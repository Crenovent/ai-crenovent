"""
Security Sandbox Service - Task 6.2.38
=======================================

Security sandbox for plugins (resource, network limits)
- Runs third-party plugins in isolated sandboxes with strict constraints
- Policy-controlled limits for CPU, memory, disk, network access
- Plugin execution policy engine with sandbox profiles
- Monitoring and kill-switch capabilities
- Backend implementation (no actual WASM/Firecracker execution - that's runtime infrastructure)

Dependencies: Task 6.2.37 (Plugin System Service)
Outputs: Security sandbox system → enables safe plugin execution isolation
"""

from typing import Dict, List, Any, Optional, Union, Tuple, Set
import json
import logging
from datetime import datetime, timezone, timedelta
from dataclasses import dataclass, field, asdict
from enum import Enum
import time
import threading
from collections import defaultdict

logger = logging.getLogger(__name__)

class SandboxRuntime(Enum):
    """Types of sandbox runtime environments"""
    WASM = "wasm"                    # WebAssembly runtime
    FIRECRACKER = "firecracker"     # Firecracker microVMs
    CONTAINER = "container"         # Container sandbox (Docker/Podman)
    PROCESS = "process"             # Process isolation
    THREAD = "thread"               # Thread-based isolation (least secure)

class ExecutionStatus(Enum):
    """Plugin execution status"""
    PENDING = "pending"
    STARTING = "starting"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    TIMEOUT = "timeout"
    KILLED = "killed"
    RESOURCE_EXCEEDED = "resource_exceeded"

class ViolationType(Enum):
    """Types of security violations"""
    RESOURCE_LIMIT_EXCEEDED = "resource_limit_exceeded"
    NETWORK_POLICY_VIOLATION = "network_policy_violation"
    FILE_ACCESS_VIOLATION = "file_access_violation"
    CAPABILITY_VIOLATION = "capability_violation"
    EXECUTION_TIME_EXCEEDED = "execution_time_exceeded"
    SUSPICIOUS_BEHAVIOR = "suspicious_behavior"

@dataclass
class ResourceUsage:
    """Current resource usage metrics"""
    cpu_percent: float = 0.0
    memory_mb: float = 0.0
    disk_mb: float = 0.0
    network_bytes_sent: int = 0
    network_bytes_received: int = 0
    file_descriptors: int = 0
    execution_time_seconds: float = 0.0
    
    # Timestamps
    last_updated: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

@dataclass
class SandboxViolation:
    """Security or resource violation record"""
    violation_id: str
    violation_type: ViolationType
    plugin_execution_id: str
    
    # Violation details
    description: str
    severity: str  # "low", "medium", "high", "critical"
    resource_type: Optional[str] = None
    limit_value: Optional[Union[int, float]] = None
    actual_value: Optional[Union[int, float]] = None
    
    # Context
    stack_trace: Optional[str] = None
    additional_context: Dict[str, Any] = field(default_factory=dict)
    
    # Response
    action_taken: str = "none"  # "none", "warning", "throttle", "kill", "quarantine"
    
    # Metadata
    detected_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

@dataclass
class SandboxPolicy:
    """Security sandbox policy"""
    policy_id: str
    policy_name: str
    sandbox_runtime: SandboxRuntime
    
    # Resource limits
    max_cpu_cores: float = 0.5
    max_memory_mb: int = 256
    max_disk_mb: int = 100
    max_execution_time_seconds: int = 30
    max_file_descriptors: int = 50
    max_network_connections: int = 5
    max_network_bandwidth_mbps: float = 1.0
    
    # Network policy
    allow_outbound_network: bool = False
    allowed_hosts: Set[str] = field(default_factory=set)
    allowed_ports: Set[int] = field(default_factory=set)
    blocked_hosts: Set[str] = field(default_factory=set)
    require_tls: bool = True
    
    # File system policy
    allowed_read_paths: Set[str] = field(default_factory=set)
    allowed_write_paths: Set[str] = field(default_factory=set)
    temp_directory_only: bool = True
    max_file_size_mb: int = 10
    
    # Capability restrictions
    allow_environment_access: bool = False
    allow_process_spawn: bool = False
    allow_system_calls: Set[str] = field(default_factory=set)
    
    # Monitoring
    enable_detailed_monitoring: bool = True
    monitoring_interval_seconds: float = 1.0
    violation_threshold_count: int = 3
    
    # Recovery
    auto_restart_on_failure: bool = False
    max_restart_attempts: int = 3
    quarantine_on_violation: bool = True
    
    # Metadata
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

@dataclass
class PluginExecution:
    """Plugin execution instance"""
    execution_id: str
    plugin_id: str
    tenant_id: str
    
    # Execution context
    sandbox_policy: SandboxPolicy
    input_data: Dict[str, Any]
    execution_parameters: Dict[str, Any] = field(default_factory=dict)
    
    # Status
    status: ExecutionStatus = ExecutionStatus.PENDING
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    
    # Resource monitoring
    resource_usage: ResourceUsage = field(default_factory=ResourceUsage)
    peak_resource_usage: ResourceUsage = field(default_factory=ResourceUsage)
    
    # Results
    output_data: Optional[Dict[str, Any]] = None
    error_message: Optional[str] = None
    exit_code: Optional[int] = None
    
    # Security
    violations: List[SandboxViolation] = field(default_factory=list)
    kill_switch_triggered: bool = False
    quarantined: bool = False
    
    # Metadata
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

@dataclass
class SandboxMetrics:
    """Sandbox system metrics"""
    total_executions: int = 0
    successful_executions: int = 0
    failed_executions: int = 0
    killed_executions: int = 0
    
    # Resource metrics
    average_cpu_usage: float = 0.0
    average_memory_usage: float = 0.0
    peak_memory_usage: float = 0.0
    average_execution_time: float = 0.0
    
    # Security metrics
    total_violations: int = 0
    violations_by_type: Dict[ViolationType, int] = field(default_factory=dict)
    quarantined_plugins: int = 0
    
    # Performance metrics
    sandbox_startup_time_ms: float = 0.0
    monitoring_overhead_percent: float = 0.0
    
    # Last updated
    last_updated: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

# Task 6.2.38: Security Sandbox Service
class SecuritySandboxService:
    """Service for managing security sandboxes for plugin execution"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
        # Sandbox policies registry
        self.sandbox_policies: Dict[str, SandboxPolicy] = {}
        
        # Active executions
        self.active_executions: Dict[str, PluginExecution] = {}  # execution_id -> execution
        self.execution_history: Dict[str, PluginExecution] = {}  # execution_id -> execution
        
        # Monitoring
        self.monitoring_threads: Dict[str, threading.Thread] = {}  # execution_id -> thread
        self.kill_switches: Dict[str, threading.Event] = {}  # execution_id -> kill event
        
        # Metrics
        self.sandbox_metrics = SandboxMetrics()
        
        # Initialize default sandbox policies
        self._initialize_default_policies()
        
        # Statistics
        self.sandbox_stats = {
            'total_policies': 0,
            'active_executions': 0,
            'total_executions': 0,
            'violations_detected': 0,
            'plugins_quarantined': 0,
            'monitoring_threads_active': 0
        }
    
    def create_sandbox_policy(self, policy_name: str, sandbox_runtime: SandboxRuntime,
                            resource_limits: Optional[Dict[str, Union[int, float]]] = None,
                            network_policy: Optional[Dict[str, Any]] = None,
                            filesystem_policy: Optional[Dict[str, Any]] = None) -> SandboxPolicy:
        """
        Create a new sandbox policy
        
        Args:
            policy_name: Name of the policy
            sandbox_runtime: Runtime environment type
            resource_limits: Resource limit overrides
            network_policy: Network policy configuration
            filesystem_policy: Filesystem policy configuration
            
        Returns:
            SandboxPolicy
        """
        policy_id = f"policy_{hash(f'{policy_name}_{datetime.now()}')}"
        
        policy = SandboxPolicy(
            policy_id=policy_id,
            policy_name=policy_name,
            sandbox_runtime=sandbox_runtime
        )
        
        # Apply resource limits
        if resource_limits:
            for limit_name, limit_value in resource_limits.items():
                if hasattr(policy, f"max_{limit_name}"):
                    setattr(policy, f"max_{limit_name}", limit_value)
        
        # Apply network policy
        if network_policy:
            policy.allow_outbound_network = network_policy.get('allow_outbound', False)
            policy.allowed_hosts = set(network_policy.get('allowed_hosts', []))
            policy.allowed_ports = set(network_policy.get('allowed_ports', []))
            policy.blocked_hosts = set(network_policy.get('blocked_hosts', []))
            policy.require_tls = network_policy.get('require_tls', True)
        
        # Apply filesystem policy
        if filesystem_policy:
            policy.allowed_read_paths = set(filesystem_policy.get('allowed_read_paths', []))
            policy.allowed_write_paths = set(filesystem_policy.get('allowed_write_paths', []))
            policy.temp_directory_only = filesystem_policy.get('temp_directory_only', True)
            policy.max_file_size_mb = filesystem_policy.get('max_file_size_mb', 10)
        
        # Store policy
        self.sandbox_policies[policy_id] = policy
        
        # Update statistics
        self._update_sandbox_stats()
        
        self.logger.info(f"✅ Created sandbox policy: {policy_name} ({sandbox_runtime.value})")
        
        return policy
    
    def execute_plugin_in_sandbox(self, plugin_id: str, tenant_id: str, 
                                 sandbox_policy_id: str, input_data: Dict[str, Any],
                                 execution_parameters: Optional[Dict[str, Any]] = None) -> PluginExecution:
        """
        Execute a plugin in a security sandbox
        
        Args:
            plugin_id: Plugin identifier
            tenant_id: Tenant identifier
            sandbox_policy_id: Sandbox policy to use
            input_data: Input data for the plugin
            execution_parameters: Additional execution parameters
            
        Returns:
            PluginExecution instance
        """
        # Get sandbox policy
        if sandbox_policy_id not in self.sandbox_policies:
            raise ValueError(f"Unknown sandbox policy: {sandbox_policy_id}")
        
        sandbox_policy = self.sandbox_policies[sandbox_policy_id]
        
        # Create execution instance
        execution_id = f"exec_{plugin_id}_{tenant_id}_{hash(f'{plugin_id}_{tenant_id}_{datetime.now()}')}"
        
        execution = PluginExecution(
            execution_id=execution_id,
            plugin_id=plugin_id,
            tenant_id=tenant_id,
            sandbox_policy=sandbox_policy,
            input_data=input_data,
            execution_parameters=execution_parameters or {}
        )
        
        # Store execution
        self.active_executions[execution_id] = execution
        
        # Start monitoring
        self._start_execution_monitoring(execution)
        
        # Simulate plugin execution (in real implementation, this would launch the actual sandbox)
        self._simulate_plugin_execution(execution)
        
        # Update statistics
        self.sandbox_stats['total_executions'] += 1
        self.sandbox_stats['active_executions'] = len(self.active_executions)
        
        self.logger.info(f"✅ Started plugin execution: {execution_id} in {sandbox_policy.sandbox_runtime.value} sandbox")
        
        return execution
    
    def kill_plugin_execution(self, execution_id: str, reason: str = "manual_kill") -> bool:
        """
        Kill a running plugin execution
        
        Args:
            execution_id: Execution identifier
            reason: Reason for killing
            
        Returns:
            True if killed successfully
        """
        if execution_id not in self.active_executions:
            self.logger.warning(f"Execution not found: {execution_id}")
            return False
        
        execution = self.active_executions[execution_id]
        
        # Trigger kill switch
        if execution_id in self.kill_switches:
            self.kill_switches[execution_id].set()
        
        # Update execution status
        execution.status = ExecutionStatus.KILLED
        execution.end_time = datetime.now(timezone.utc)
        execution.kill_switch_triggered = True
        execution.error_message = f"Execution killed: {reason}"
        
        # Stop monitoring
        self._stop_execution_monitoring(execution_id)
        
        # Move to history
        self.execution_history[execution_id] = execution
        del self.active_executions[execution_id]
        
        # Update statistics
        self.sandbox_stats['killed_executions'] += 1
        self.sandbox_stats['active_executions'] = len(self.active_executions)
        
        self.logger.info(f"✅ Killed plugin execution: {execution_id} ({reason})")
        
        return True
    
    def get_execution_status(self, execution_id: str) -> Optional[PluginExecution]:
        """
        Get execution status
        
        Args:
            execution_id: Execution identifier
            
        Returns:
            PluginExecution or None if not found
        """
        # Check active executions first
        if execution_id in self.active_executions:
            return self.active_executions[execution_id]
        
        # Check history
        if execution_id in self.execution_history:
            return self.execution_history[execution_id]
        
        return None
    
    def quarantine_plugin(self, plugin_id: str, tenant_id: str, reason: str) -> bool:
        """
        Quarantine a plugin due to security violations
        
        Args:
            plugin_id: Plugin identifier
            tenant_id: Tenant identifier
            reason: Reason for quarantine
            
        Returns:
            True if quarantined successfully
        """
        # Find all active executions for this plugin
        executions_to_kill = []
        for execution_id, execution in self.active_executions.items():
            if execution.plugin_id == plugin_id and execution.tenant_id == tenant_id:
                executions_to_kill.append(execution_id)
        
        # Kill all active executions
        for execution_id in executions_to_kill:
            self.kill_plugin_execution(execution_id, f"quarantine: {reason}")
            execution = self.execution_history[execution_id]
            execution.quarantined = True
        
        # Update statistics
        self.sandbox_stats['plugins_quarantined'] += 1
        
        self.logger.warning(f"🔒 Quarantined plugin: {plugin_id} for tenant {tenant_id} ({reason})")
        
        return True
    
    def get_sandbox_metrics(self) -> SandboxMetrics:
        """Get current sandbox metrics"""
        # Update metrics from current state
        self.sandbox_metrics.total_executions = self.sandbox_stats['total_executions']
        self.sandbox_metrics.quarantined_plugins = self.sandbox_stats['plugins_quarantined']
        
        # Calculate averages from execution history
        if self.execution_history:
            completed_executions = [e for e in self.execution_history.values() if e.status == ExecutionStatus.COMPLETED]
            
            if completed_executions:
                self.sandbox_metrics.average_cpu_usage = sum(e.peak_resource_usage.cpu_percent for e in completed_executions) / len(completed_executions)
                self.sandbox_metrics.average_memory_usage = sum(e.peak_resource_usage.memory_mb for e in completed_executions) / len(completed_executions)
                self.sandbox_metrics.peak_memory_usage = max(e.peak_resource_usage.memory_mb for e in completed_executions)
                self.sandbox_metrics.average_execution_time = sum(e.peak_resource_usage.execution_time_seconds for e in completed_executions) / len(completed_executions)
        
        # Count violations by type
        all_violations = []
        for execution in list(self.active_executions.values()) + list(self.execution_history.values()):
            all_violations.extend(execution.violations)
        
        self.sandbox_metrics.total_violations = len(all_violations)
        self.sandbox_metrics.violations_by_type = defaultdict(int)
        for violation in all_violations:
            self.sandbox_metrics.violations_by_type[violation.violation_type] += 1
        
        self.sandbox_metrics.last_updated = datetime.now(timezone.utc)
        
        return self.sandbox_metrics
    
    def validate_sandbox_policy(self, policy: SandboxPolicy) -> List[str]:
        """
        Validate a sandbox policy for security and consistency
        
        Args:
            policy: Sandbox policy to validate
            
        Returns:
            List of validation issues
        """
        issues = []
        
        # Check resource limits
        if policy.max_cpu_cores <= 0:
            issues.append("CPU cores limit must be positive")
        
        if policy.max_memory_mb <= 0:
            issues.append("Memory limit must be positive")
        
        if policy.max_execution_time_seconds <= 0:
            issues.append("Execution time limit must be positive")
        
        # Check for overly permissive settings
        if policy.max_cpu_cores > 4:
            issues.append("CPU cores limit is very high (> 4 cores)")
        
        if policy.max_memory_mb > 2048:
            issues.append("Memory limit is very high (> 2GB)")
        
        if policy.max_execution_time_seconds > 600:
            issues.append("Execution time limit is very high (> 10 minutes)")
        
        # Check network policy consistency
        if policy.allow_outbound_network and not policy.allowed_hosts and not policy.allowed_ports:
            issues.append("Outbound network allowed but no hosts or ports specified")
        
        # Check filesystem policy
        if not policy.temp_directory_only and not policy.allowed_write_paths:
            issues.append("Temp directory restriction disabled but no write paths specified")
        
        # Check monitoring settings
        if policy.monitoring_interval_seconds < 0.1:
            issues.append("Monitoring interval too low (< 0.1 seconds)")
        
        if policy.violation_threshold_count <= 0:
            issues.append("Violation threshold must be positive")
        
        return issues
    
    def get_execution_logs(self, execution_id: str, include_violations: bool = True) -> Dict[str, Any]:
        """
        Get detailed execution logs
        
        Args:
            execution_id: Execution identifier
            include_violations: Whether to include violation details
            
        Returns:
            Dictionary with execution logs
        """
        execution = self.get_execution_status(execution_id)
        if not execution:
            return {'error': 'Execution not found'}
        
        logs = {
            'execution_id': execution.execution_id,
            'plugin_id': execution.plugin_id,
            'tenant_id': execution.tenant_id,
            'status': execution.status.value,
            'start_time': execution.start_time.isoformat() if execution.start_time else None,
            'end_time': execution.end_time.isoformat() if execution.end_time else None,
            'duration_seconds': (execution.end_time - execution.start_time).total_seconds() if execution.start_time and execution.end_time else None,
            'resource_usage': asdict(execution.resource_usage),
            'peak_resource_usage': asdict(execution.peak_resource_usage),
            'kill_switch_triggered': execution.kill_switch_triggered,
            'quarantined': execution.quarantined,
            'error_message': execution.error_message,
            'exit_code': execution.exit_code
        }
        
        if include_violations:
            logs['violations'] = [asdict(violation) for violation in execution.violations]
            logs['violation_count'] = len(execution.violations)
        
        return logs
    
    def get_sandbox_statistics(self) -> Dict[str, Any]:
        """Get sandbox service statistics"""
        return {
            **self.sandbox_stats,
            'sandbox_policies': len(self.sandbox_policies),
            'execution_history_size': len(self.execution_history),
            'monitoring_threads_active': len(self.monitoring_threads)
        }
    
    def _initialize_default_policies(self):
        """Initialize default sandbox policies"""
        
        # Minimal security policy
        minimal_policy = self.create_sandbox_policy(
            "minimal_security",
            SandboxRuntime.WASM,
            resource_limits={
                'cpu_cores': 0.1,
                'memory_mb': 64,
                'disk_mb': 10,
                'execution_time_seconds': 5
            },
            network_policy={'allow_outbound': False},
            filesystem_policy={'temp_directory_only': True}
        )
        
        # Standard policy
        standard_policy = self.create_sandbox_policy(
            "standard_security",
            SandboxRuntime.CONTAINER,
            resource_limits={
                'cpu_cores': 0.5,
                'memory_mb': 256,
                'disk_mb': 100,
                'execution_time_seconds': 30
            },
            network_policy={
                'allow_outbound': True,
                'allowed_hosts': ['api.example.com'],
                'allowed_ports': [80, 443]
            }
        )
        
        # High performance policy
        performance_policy = self.create_sandbox_policy(
            "high_performance",
            SandboxRuntime.FIRECRACKER,
            resource_limits={
                'cpu_cores': 2.0,
                'memory_mb': 1024,
                'disk_mb': 500,
                'execution_time_seconds': 120
            },
            network_policy={
                'allow_outbound': True,
                'require_tls': True
            }
        )
        
        self.logger.info(f"✅ Initialized {len(self.sandbox_policies)} default sandbox policies")
    
    def _start_execution_monitoring(self, execution: PluginExecution):
        """Start monitoring thread for execution"""
        execution_id = execution.execution_id
        
        # Create kill switch
        kill_switch = threading.Event()
        self.kill_switches[execution_id] = kill_switch
        
        # Start monitoring thread
        monitor_thread = threading.Thread(
            target=self._monitor_execution,
            args=(execution, kill_switch),
            daemon=True
        )
        monitor_thread.start()
        self.monitoring_threads[execution_id] = monitor_thread
        
        self.sandbox_stats['monitoring_threads_active'] = len(self.monitoring_threads)
    
    def _stop_execution_monitoring(self, execution_id: str):
        """Stop monitoring thread for execution"""
        if execution_id in self.kill_switches:
            self.kill_switches[execution_id].set()
            del self.kill_switches[execution_id]
        
        if execution_id in self.monitoring_threads:
            # Thread will stop when kill switch is set
            del self.monitoring_threads[execution_id]
        
        self.sandbox_stats['monitoring_threads_active'] = len(self.monitoring_threads)
    
    def _monitor_execution(self, execution: PluginExecution, kill_switch: threading.Event):
        """Monitor execution for resource usage and violations"""
        policy = execution.sandbox_policy
        start_time = time.time()
        
        while not kill_switch.is_set() and execution.status == ExecutionStatus.RUNNING:
            current_time = time.time()
            execution_time = current_time - start_time
            
            # Simulate resource usage (in real implementation, this would query actual metrics)
            execution.resource_usage.cpu_percent = min(50.0, execution_time * 2)  # Simulate increasing CPU
            execution.resource_usage.memory_mb = min(200.0, execution_time * 5)   # Simulate increasing memory
            execution.resource_usage.execution_time_seconds = execution_time
            execution.resource_usage.last_updated = datetime.now(timezone.utc)
            
            # Update peak usage
            if execution.resource_usage.cpu_percent > execution.peak_resource_usage.cpu_percent:
                execution.peak_resource_usage.cpu_percent = execution.resource_usage.cpu_percent
            if execution.resource_usage.memory_mb > execution.peak_resource_usage.memory_mb:
                execution.peak_resource_usage.memory_mb = execution.resource_usage.memory_mb
            execution.peak_resource_usage.execution_time_seconds = execution_time
            
            # Check for violations
            violations = self._check_resource_violations(execution, policy)
            for violation in violations:
                execution.violations.append(violation)
                self._handle_violation(execution, violation)
            
            # Check execution time limit
            if execution_time > policy.max_execution_time_seconds:
                timeout_violation = SandboxViolation(
                    violation_id=f"timeout_{execution.execution_id}_{len(execution.violations)}",
                    violation_type=ViolationType.EXECUTION_TIME_EXCEEDED,
                    plugin_execution_id=execution.execution_id,
                    description=f"Execution time exceeded limit: {execution_time:.1f}s > {policy.max_execution_time_seconds}s",
                    severity="high",
                    resource_type="execution_time",
                    limit_value=policy.max_execution_time_seconds,
                    actual_value=execution_time,
                    action_taken="kill"
                )
                execution.violations.append(timeout_violation)
                
                # Kill execution
                execution.status = ExecutionStatus.TIMEOUT
                execution.end_time = datetime.now(timezone.utc)
                kill_switch.set()
                break
            
            # Sleep for monitoring interval
            time.sleep(policy.monitoring_interval_seconds)
    
    def _check_resource_violations(self, execution: PluginExecution, policy: SandboxPolicy) -> List[SandboxViolation]:
        """Check for resource limit violations"""
        violations = []
        
        # Check CPU limit
        if execution.resource_usage.cpu_percent > (policy.max_cpu_cores * 100):
            violation = SandboxViolation(
                violation_id=f"cpu_{execution.execution_id}_{len(execution.violations)}",
                violation_type=ViolationType.RESOURCE_LIMIT_EXCEEDED,
                plugin_execution_id=execution.execution_id,
                description=f"CPU usage exceeded limit: {execution.resource_usage.cpu_percent:.1f}% > {policy.max_cpu_cores * 100}%",
                severity="medium",
                resource_type="cpu",
                limit_value=policy.max_cpu_cores * 100,
                actual_value=execution.resource_usage.cpu_percent,
                action_taken="throttle"
            )
            violations.append(violation)
        
        # Check memory limit
        if execution.resource_usage.memory_mb > policy.max_memory_mb:
            violation = SandboxViolation(
                violation_id=f"memory_{execution.execution_id}_{len(execution.violations)}",
                violation_type=ViolationType.RESOURCE_LIMIT_EXCEEDED,
                plugin_execution_id=execution.execution_id,
                description=f"Memory usage exceeded limit: {execution.resource_usage.memory_mb:.1f}MB > {policy.max_memory_mb}MB",
                severity="high",
                resource_type="memory",
                limit_value=policy.max_memory_mb,
                actual_value=execution.resource_usage.memory_mb,
                action_taken="kill" if execution.resource_usage.memory_mb > policy.max_memory_mb * 1.5 else "warning"
            )
            violations.append(violation)
        
        return violations
    
    def _handle_violation(self, execution: PluginExecution, violation: SandboxViolation):
        """Handle a security or resource violation"""
        self.sandbox_stats['violations_detected'] += 1
        
        # Log violation
        self.logger.warning(f"🚨 Sandbox violation: {violation.description}")
        
        # Take action based on severity and type
        if violation.action_taken == "kill":
            self.kill_plugin_execution(execution.execution_id, f"violation: {violation.description}")
        elif violation.action_taken == "quarantine":
            self.quarantine_plugin(execution.plugin_id, execution.tenant_id, violation.description)
        
        # Check violation threshold
        policy = execution.sandbox_policy
        if len(execution.violations) >= policy.violation_threshold_count:
            if policy.quarantine_on_violation:
                self.quarantine_plugin(execution.plugin_id, execution.tenant_id, 
                                     f"violation threshold exceeded: {len(execution.violations)} violations")
    
    def _simulate_plugin_execution(self, execution: PluginExecution):
        """Simulate plugin execution (placeholder for actual sandbox execution)"""
        execution.status = ExecutionStatus.RUNNING
        execution.start_time = datetime.now(timezone.utc)
        
        # In a real implementation, this would:
        # 1. Launch the plugin in the specified sandbox runtime
        # 2. Pass input data to the plugin
        # 3. Monitor resource usage and security constraints
        # 4. Collect output data
        # 5. Clean up sandbox environment
        
        # For simulation, we'll complete after a short delay
        def complete_execution():
            time.sleep(2)  # Simulate execution time
            if execution.status == ExecutionStatus.RUNNING:
                execution.status = ExecutionStatus.COMPLETED
                execution.end_time = datetime.now(timezone.utc)
                execution.output_data = {
                    "result": "Plugin execution completed",
                    "input_processed": execution.input_data,
                    "execution_time": (execution.end_time - execution.start_time).total_seconds()
                }
                
                # Move to history
                self.execution_history[execution.execution_id] = execution
                if execution.execution_id in self.active_executions:
                    del self.active_executions[execution.execution_id]
                
                # Stop monitoring
                self._stop_execution_monitoring(execution.execution_id)
                
                # Update statistics
                self.sandbox_stats['active_executions'] = len(self.active_executions)
        
        # Start completion thread
        completion_thread = threading.Thread(target=complete_execution, daemon=True)
        completion_thread.start()
    
    def _update_sandbox_stats(self):
        """Update sandbox statistics"""
        self.sandbox_stats['total_policies'] = len(self.sandbox_policies)

# API Interface
class SecuritySandboxAPI:
    """API interface for security sandbox operations"""
    
    def __init__(self, sandbox_service: Optional[SecuritySandboxService] = None):
        self.sandbox_service = sandbox_service or SecuritySandboxService()
    
    def execute_plugin(self, plugin_id: str, tenant_id: str, sandbox_policy_id: str,
                      input_data: Dict[str, Any]) -> Dict[str, Any]:
        """API endpoint to execute plugin in sandbox"""
        try:
            execution = self.sandbox_service.execute_plugin_in_sandbox(
                plugin_id, tenant_id, sandbox_policy_id, input_data
            )
            
            return {
                'success': True,
                'execution_id': execution.execution_id,
                'status': execution.status.value,
                'sandbox_runtime': execution.sandbox_policy.sandbox_runtime.value
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': str(e),
                'error_type': type(e).__name__
            }
    
    def get_execution_status(self, execution_id: str) -> Dict[str, Any]:
        """API endpoint to get execution status"""
        execution = self.sandbox_service.get_execution_status(execution_id)
        
        if execution:
            return {
                'success': True,
                'execution': asdict(execution)
            }
        else:
            return {
                'success': False,
                'error': 'Execution not found'
            }
    
    def kill_execution(self, execution_id: str, reason: str = "manual_kill") -> Dict[str, Any]:
        """API endpoint to kill execution"""
        success = self.sandbox_service.kill_plugin_execution(execution_id, reason)
        
        return {
            'success': success,
            'execution_id': execution_id,
            'reason': reason
        }

# Test Functions
def run_security_sandbox_tests():
    """Run comprehensive security sandbox tests"""
    print("=== Security Sandbox Service Tests ===")
    
    # Initialize service
    sandbox_service = SecuritySandboxService()
    sandbox_api = SecuritySandboxAPI(sandbox_service)
    
    # Test 1: Default policies initialization
    print("\n1. Testing default policies initialization...")
    stats = sandbox_service.get_sandbox_statistics()
    print(f"   Default policies created: {stats['sandbox_policies']}")
    
    for policy_id, policy in sandbox_service.sandbox_policies.items():
        print(f"     - {policy.policy_name}: {policy.sandbox_runtime.value} runtime")
        print(f"       CPU: {policy.max_cpu_cores} cores, Memory: {policy.max_memory_mb}MB")
    
    # Test 2: Custom sandbox policy creation
    print("\n2. Testing custom sandbox policy creation...")
    
    custom_policy = sandbox_service.create_sandbox_policy(
        "test_policy",
        SandboxRuntime.CONTAINER,
        resource_limits={
            'cpu_cores': 1.0,
            'memory_mb': 512,
            'execution_time_seconds': 60
        },
        network_policy={
            'allow_outbound': True,
            'allowed_hosts': ['api.test.com'],
            'require_tls': True
        }
    )
    
    print(f"   Custom policy created: {custom_policy.policy_name}")
    print(f"   Runtime: {custom_policy.sandbox_runtime.value}")
    print(f"   Max CPU: {custom_policy.max_cpu_cores} cores")
    print(f"   Max Memory: {custom_policy.max_memory_mb}MB")
    print(f"   Network allowed: {custom_policy.allow_outbound_network}")
    
    # Test 3: Policy validation
    print("\n3. Testing sandbox policy validation...")
    
    # Valid policy
    validation_issues = sandbox_service.validate_sandbox_policy(custom_policy)
    print(f"   Custom policy validation: {'✅ PASS' if not validation_issues else '❌ FAIL'}")
    if validation_issues:
        for issue in validation_issues:
            print(f"     - {issue}")
    
    # Invalid policy
    invalid_policy = SandboxPolicy(
        policy_id="invalid",
        policy_name="Invalid Policy",
        sandbox_runtime=SandboxRuntime.WASM,
        max_cpu_cores=-1,  # Invalid
        max_memory_mb=4096,  # Too high
        max_execution_time_seconds=0  # Invalid
    )
    
    validation_issues = sandbox_service.validate_sandbox_policy(invalid_policy)
    print(f"   Invalid policy validation: {'❌ FAIL' if validation_issues else '✅ UNEXPECTED PASS'}")
    print(f"   Validation issues found: {len(validation_issues)}")
    
    # Test 4: Plugin execution in sandbox
    print("\n4. Testing plugin execution in sandbox...")
    
    # Get a policy to use
    policy_id = list(sandbox_service.sandbox_policies.keys())[0]
    
    execution = sandbox_service.execute_plugin_in_sandbox(
        plugin_id="test_plugin",
        tenant_id="test_tenant",
        sandbox_policy_id=policy_id,
        input_data={"test": "data", "value": 42}
    )
    
    print(f"   Plugin execution started: {execution.execution_id}")
    print(f"   Status: {execution.status.value}")
    print(f"   Sandbox runtime: {execution.sandbox_policy.sandbox_runtime.value}")
    
    # Wait a moment for execution to progress
    time.sleep(3)
    
    # Check execution status
    updated_execution = sandbox_service.get_execution_status(execution.execution_id)
    if updated_execution:
        print(f"   Updated status: {updated_execution.status.value}")
        print(f"   Resource usage - CPU: {updated_execution.resource_usage.cpu_percent:.1f}%, Memory: {updated_execution.resource_usage.memory_mb:.1f}MB")
        print(f"   Violations: {len(updated_execution.violations)}")
    
    # Test 5: Resource violation simulation
    print("\n5. Testing resource violation simulation...")
    
    # Create a policy with very low limits
    strict_policy = sandbox_service.create_sandbox_policy(
        "strict_test_policy",
        SandboxRuntime.WASM,
        resource_limits={
            'cpu_cores': 0.01,  # Very low CPU limit
            'memory_mb': 10,    # Very low memory limit
            'execution_time_seconds': 1  # Very short time limit
        }
    )
    
    strict_execution = sandbox_service.execute_plugin_in_sandbox(
        plugin_id="resource_heavy_plugin",
        tenant_id="test_tenant",
        sandbox_policy_id=strict_policy.policy_id,
        input_data={"heavy_computation": True}
    )
    
    print(f"   Strict execution started: {strict_execution.execution_id}")
    
    # Wait for violations to occur
    time.sleep(2)
    
    updated_strict_execution = sandbox_service.get_execution_status(strict_execution.execution_id)
    if updated_strict_execution:
        print(f"   Strict execution status: {updated_strict_execution.status.value}")
        print(f"   Violations detected: {len(updated_strict_execution.violations)}")
        for violation in updated_strict_execution.violations:
            print(f"     - {violation.violation_type.value}: {violation.description}")
    
    # Test 6: Kill switch functionality
    print("\n6. Testing kill switch functionality...")
    
    # Start another execution
    kill_test_execution = sandbox_service.execute_plugin_in_sandbox(
        plugin_id="long_running_plugin",
        tenant_id="test_tenant",
        sandbox_policy_id=policy_id,
        input_data={"duration": "long"}
    )
    
    print(f"   Long running execution started: {kill_test_execution.execution_id}")
    
    # Kill it immediately
    killed = sandbox_service.kill_plugin_execution(kill_test_execution.execution_id, "test_kill")
    print(f"   Kill execution: {'✅ PASS' if killed else '❌ FAIL'}")
    
    if killed:
        killed_execution = sandbox_service.get_execution_status(kill_test_execution.execution_id)
        if killed_execution:
            print(f"   Killed execution status: {killed_execution.status.value}")
            print(f"   Kill switch triggered: {killed_execution.kill_switch_triggered}")
    
    # Test 7: Plugin quarantine
    print("\n7. Testing plugin quarantine...")
    
    quarantined = sandbox_service.quarantine_plugin(
        "malicious_plugin", "test_tenant", "suspicious behavior detected"
    )
    print(f"   Plugin quarantine: {'✅ PASS' if quarantined else '❌ FAIL'}")
    
    # Test 8: Execution logs
    print("\n8. Testing execution logs...")
    
    if updated_execution:
        logs = sandbox_service.get_execution_logs(updated_execution.execution_id)
        print(f"   Execution logs retrieved: {'✅ PASS' if 'error' not in logs else '❌ FAIL'}")
        if 'error' not in logs:
            print(f"   Log entries: {len(logs)}")
            print(f"   Status: {logs.get('status')}")
            print(f"   Duration: {logs.get('duration_seconds', 'N/A')} seconds")
            print(f"   Violations: {logs.get('violation_count', 0)}")
    
    # Test 9: Sandbox metrics
    print("\n9. Testing sandbox metrics...")
    
    metrics = sandbox_service.get_sandbox_metrics()
    print(f"   Sandbox metrics:")
    print(f"     Total executions: {metrics.total_executions}")
    print(f"     Total violations: {metrics.total_violations}")
    print(f"     Quarantined plugins: {metrics.quarantined_plugins}")
    print(f"     Average CPU usage: {metrics.average_cpu_usage:.1f}%")
    print(f"     Average memory usage: {metrics.average_memory_usage:.1f}MB")
    print(f"     Peak memory usage: {metrics.peak_memory_usage:.1f}MB")
    
    # Test 10: API interface
    print("\n10. Testing API interface...")
    
    # Test API execution
    api_execution_result = sandbox_api.execute_plugin(
        "api_test_plugin", "test_tenant", policy_id, {"api_test": True}
    )
    print(f"   API execution: {'✅ PASS' if api_execution_result['success'] else '❌ FAIL'}")
    
    if api_execution_result['success']:
        api_execution_id = api_execution_result['execution_id']
        
        # Test API status check
        api_status_result = sandbox_api.get_execution_status(api_execution_id)
        print(f"   API status check: {'✅ PASS' if api_status_result['success'] else '❌ FAIL'}")
        
        # Test API kill
        api_kill_result = sandbox_api.kill_execution(api_execution_id, "api_test_kill")
        print(f"   API kill: {'✅ PASS' if api_kill_result['success'] else '❌ FAIL'}")
    
    # Test 11: Statistics
    print("\n11. Testing statistics...")
    
    final_stats = sandbox_service.get_sandbox_statistics()
    print(f"   Total policies: {final_stats['sandbox_policies']}")
    print(f"   Total executions: {final_stats['total_executions']}")
    print(f"   Active executions: {final_stats['active_executions']}")
    print(f"   Violations detected: {final_stats['violations_detected']}")
    print(f"   Plugins quarantined: {final_stats['plugins_quarantined']}")
    print(f"   Monitoring threads: {final_stats['monitoring_threads_active']}")
    
    print(f"\n=== Test Summary ===")
    print(f"Security sandbox service tested successfully")
    print(f"Sandbox policies: {final_stats['sandbox_policies']}")
    print(f"Total executions: {final_stats['total_executions']}")
    print(f"Violations detected: {final_stats['violations_detected']}")
    print(f"Plugins quarantined: {final_stats['plugins_quarantined']}")
    
    return sandbox_service, sandbox_api

if __name__ == "__main__":
    run_security_sandbox_tests()



