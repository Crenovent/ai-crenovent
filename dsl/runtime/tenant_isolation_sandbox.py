# Tenant Isolation Sandbox
# Tasks 6.5-T03, T09, T10: Tenant-aware execution sandbox, quotas, throttling

import asyncio
import json
import uuid
import time
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Any, Optional, Union, Callable
from dataclasses import dataclass, asdict
from enum import Enum
import logging
import threading
from collections import defaultdict, deque

logger = logging.getLogger(__name__)

class IsolationLevel(Enum):
    """Levels of tenant isolation"""
    LOGICAL = "logical"  # Shared resources with logical separation
    PROCESS = "process"  # Separate processes per tenant
    CONTAINER = "container"  # Separate containers per tenant
    NAMESPACE = "namespace"  # Kubernetes namespace isolation

class ResourceType(Enum):
    """Types of resources to monitor and limit"""
    CPU = "cpu"
    MEMORY = "memory"
    STORAGE = "storage"
    NETWORK = "network"
    WORKFLOWS = "workflows"
    API_CALLS = "api_calls"

@dataclass
class ResourceUsage:
    """Current resource usage for a tenant"""
    tenant_id: int
    resource_type: ResourceType
    current_usage: float
    peak_usage: float
    average_usage: float
    usage_history: List[float]
    last_updated: str

@dataclass
class TenantSandbox:
    """Isolated execution sandbox for a tenant"""
    tenant_id: int
    sandbox_id: str
    isolation_level: IsolationLevel
    namespace: str
    resource_limits: Dict[ResourceType, float]
    current_usage: Dict[ResourceType, ResourceUsage]
    active_workflows: Dict[str, Dict[str, Any]]
    created_at: str
    last_activity: str
    status: str  # active, suspended, terminated

class TenantThrottler:
    """
    Tenant-level throttling and rate limiting
    Task 6.5-T10: Implement throttling at tenant boundary
    """
    
    def __init__(self):
        self.rate_limits: Dict[int, Dict[str, Any]] = {}
        self.usage_windows: Dict[int, Dict[str, deque]] = defaultdict(lambda: defaultdict(deque))
        self._lock = threading.RLock()
    
    def set_rate_limit(
        self,
        tenant_id: int,
        resource_type: str,
        limit: int,
        window_seconds: int = 60
    ) -> None:
        """Set rate limit for tenant and resource type"""
        with self._lock:
            if tenant_id not in self.rate_limits:
                self.rate_limits[tenant_id] = {}
            
            self.rate_limits[tenant_id][resource_type] = {
                'limit': limit,
                'window_seconds': window_seconds,
                'current_count': 0,
                'window_start': time.time()
            }
    
    def check_rate_limit(
        self,
        tenant_id: int,
        resource_type: str,
        increment: int = 1
    ) -> Tuple[bool, Dict[str, Any]]:
        """Check if request is within rate limit"""
        
        with self._lock:
            if tenant_id not in self.rate_limits or resource_type not in self.rate_limits[tenant_id]:
                return True, {'allowed': True, 'reason': 'no_limit_set'}
            
            limit_config = self.rate_limits[tenant_id][resource_type]
            current_time = time.time()
            
            # Check if window has expired
            if current_time - limit_config['window_start'] >= limit_config['window_seconds']:
                # Reset window
                limit_config['current_count'] = 0
                limit_config['window_start'] = current_time
            
            # Check if adding increment would exceed limit
            if limit_config['current_count'] + increment > limit_config['limit']:
                remaining_time = limit_config['window_seconds'] - (current_time - limit_config['window_start'])
                return False, {
                    'allowed': False,
                    'reason': 'rate_limit_exceeded',
                    'limit': limit_config['limit'],
                    'current_usage': limit_config['current_count'],
                    'reset_in_seconds': remaining_time
                }
            
            # Update usage
            limit_config['current_count'] += increment
            
            return True, {
                'allowed': True,
                'limit': limit_config['limit'],
                'current_usage': limit_config['current_count'],
                'remaining': limit_config['limit'] - limit_config['current_count']
            }

class TenantIsolationSandbox:
    """
    Comprehensive tenant isolation and sandboxing system
    Tasks 6.5-T03, T09, T10: Sandbox, quotas, throttling
    """
    
    def __init__(self, tenant_context_manager=None):
        self.tenant_context_manager = tenant_context_manager
        self.sandboxes: Dict[int, TenantSandbox] = {}
        self.throttler = TenantThrottler()
        self.resource_monitors: Dict[int, Dict[ResourceType, ResourceUsage]] = defaultdict(dict)
        self._sandbox_lock = threading.RLock()
        
        # Default resource limits by tenant tier
        self.default_resource_limits = {
            'starter': {
                ResourceType.CPU: 2.0,  # CPU cores
                ResourceType.MEMORY: 4.0,  # GB
                ResourceType.STORAGE: 10.0,  # GB
                ResourceType.NETWORK: 100.0,  # Mbps
                ResourceType.WORKFLOWS: 5.0,  # Concurrent workflows
                ResourceType.API_CALLS: 100.0  # Per minute
            },
            'professional': {
                ResourceType.CPU: 8.0,
                ResourceType.MEMORY: 16.0,
                ResourceType.STORAGE: 100.0,
                ResourceType.NETWORK: 500.0,
                ResourceType.WORKFLOWS: 25.0,
                ResourceType.API_CALLS: 1000.0
            },
            'enterprise': {
                ResourceType.CPU: 32.0,
                ResourceType.MEMORY: 64.0,
                ResourceType.STORAGE: 1000.0,
                ResourceType.NETWORK: 2000.0,
                ResourceType.WORKFLOWS: 100.0,
                ResourceType.API_CALLS: 10000.0
            },
            'enterprise_plus': {
                ResourceType.CPU: 128.0,
                ResourceType.MEMORY: 256.0,
                ResourceType.STORAGE: 10000.0,
                ResourceType.NETWORK: 10000.0,
                ResourceType.WORKFLOWS: 500.0,
                ResourceType.API_CALLS: 100000.0
            }
        }
    
    async def create_tenant_sandbox(
        self,
        tenant_id: int,
        isolation_level: IsolationLevel = IsolationLevel.LOGICAL
    ) -> TenantSandbox:
        """Create isolated sandbox for tenant"""
        
        # Get tenant context for resource limits
        tenant_context = None
        if self.tenant_context_manager:
            tenant_context = await self.tenant_context_manager.get_tenant_context(tenant_id)
        
        # Determine resource limits
        if tenant_context:
            tier_key = tenant_context.tenant_tier.value
            resource_limits = self.default_resource_limits.get(tier_key, self.default_resource_limits['professional'])
        else:
            resource_limits = self.default_resource_limits['professional']
        
        # Create sandbox
        sandbox = TenantSandbox(
            tenant_id=tenant_id,
            sandbox_id=str(uuid.uuid4()),
            isolation_level=isolation_level,
            namespace=f"tenant-{tenant_id}",
            resource_limits=resource_limits,
            current_usage={},
            active_workflows={},
            created_at=datetime.now(timezone.utc).isoformat(),
            last_activity=datetime.now(timezone.utc).isoformat(),
            status="active"
        )
        
        # Initialize resource usage tracking
        for resource_type in ResourceType:
            sandbox.current_usage[resource_type] = ResourceUsage(
                tenant_id=tenant_id,
                resource_type=resource_type,
                current_usage=0.0,
                peak_usage=0.0,
                average_usage=0.0,
                usage_history=[],
                last_updated=datetime.now(timezone.utc).isoformat()
            )
        
        # Set up throttling
        if tenant_context:
            self.throttler.set_rate_limit(
                tenant_id,
                'api_calls',
                tenant_context.quotas.max_api_calls_per_minute,
                60
            )
            self.throttler.set_rate_limit(
                tenant_id,
                'workflows',
                tenant_context.quotas.max_workflows_per_hour,
                3600
            )
        
        # Store sandbox
        with self._sandbox_lock:
            self.sandboxes[tenant_id] = sandbox
        
        logger.info(f"Created tenant sandbox: {tenant_id} ({isolation_level.value})")
        return sandbox
    
    async def get_tenant_sandbox(self, tenant_id: int) -> Optional[TenantSandbox]:
        """Get existing tenant sandbox"""
        with self._sandbox_lock:
            return self.sandboxes.get(tenant_id)
    
    async def execute_in_sandbox(
        self,
        tenant_id: int,
        workflow_id: str,
        execution_function: Callable,
        execution_args: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Execute workflow in tenant sandbox with resource monitoring"""
        
        # Get or create sandbox
        sandbox = await self.get_tenant_sandbox(tenant_id)
        if not sandbox:
            sandbox = await self.create_tenant_sandbox(tenant_id)
        
        # Check rate limits
        workflow_allowed, workflow_limit_info = self.throttler.check_rate_limit(
            tenant_id, 'workflows', 1
        )
        
        if not workflow_allowed:
            return {
                'success': False,
                'error': 'Rate limit exceeded',
                'limit_info': workflow_limit_info,
                'tenant_id': tenant_id
            }
        
        # Check resource availability
        resource_check = await self._check_resource_availability(sandbox, ResourceType.WORKFLOWS, 1)
        if not resource_check['available']:
            return {
                'success': False,
                'error': 'Resource limit exceeded',
                'resource_info': resource_check,
                'tenant_id': tenant_id
            }
        
        execution_id = str(uuid.uuid4())
        start_time = time.time()
        
        try:
            # Track workflow start
            await self._track_workflow_start(sandbox, workflow_id, execution_id)
            
            # Execute with resource monitoring
            result = await self._execute_with_monitoring(
                sandbox,
                execution_function,
                execution_args
            )
            
            execution_time = time.time() - start_time
            
            # Track successful completion
            await self._track_workflow_completion(
                sandbox,
                workflow_id,
                execution_id,
                execution_time,
                True
            )
            
            return {
                'success': True,
                'result': result,
                'execution_id': execution_id,
                'execution_time_seconds': execution_time,
                'tenant_id': tenant_id,
                'sandbox_id': sandbox.sandbox_id
            }
            
        except Exception as e:
            execution_time = time.time() - start_time
            
            # Track failed completion
            await self._track_workflow_completion(
                sandbox,
                workflow_id,
                execution_id,
                execution_time,
                False
            )
            
            logger.error(f"Workflow execution failed in sandbox {sandbox.sandbox_id}: {e}")
            
            return {
                'success': False,
                'error': str(e),
                'execution_id': execution_id,
                'execution_time_seconds': execution_time,
                'tenant_id': tenant_id,
                'sandbox_id': sandbox.sandbox_id
            }
    
    async def _check_resource_availability(
        self,
        sandbox: TenantSandbox,
        resource_type: ResourceType,
        required_amount: float
    ) -> Dict[str, Any]:
        """Check if required resources are available"""
        
        current_usage = sandbox.current_usage[resource_type].current_usage
        resource_limit = sandbox.resource_limits[resource_type]
        
        available_amount = resource_limit - current_usage
        
        if required_amount > available_amount:
            return {
                'available': False,
                'required': required_amount,
                'available_amount': available_amount,
                'current_usage': current_usage,
                'limit': resource_limit,
                'utilization_percent': (current_usage / resource_limit) * 100
            }
        
        return {
            'available': True,
            'required': required_amount,
            'available_amount': available_amount,
            'current_usage': current_usage,
            'limit': resource_limit,
            'utilization_percent': (current_usage / resource_limit) * 100
        }
    
    async def _execute_with_monitoring(
        self,
        sandbox: TenantSandbox,
        execution_function: Callable,
        execution_args: Dict[str, Any]
    ) -> Any:
        """Execute function with resource monitoring"""
        
        # Start resource monitoring
        monitoring_task = asyncio.create_task(
            self._monitor_execution_resources(sandbox)
        )
        
        try:
            # Execute the function
            if asyncio.iscoroutinefunction(execution_function):
                result = await execution_function(**execution_args)
            else:
                result = execution_function(**execution_args)
            
            return result
            
        finally:
            # Stop monitoring
            monitoring_task.cancel()
            try:
                await monitoring_task
            except asyncio.CancelledError:
                pass
    
    async def _monitor_execution_resources(self, sandbox: TenantSandbox) -> None:
        """Monitor resource usage during execution"""
        
        while True:
            try:
                # Simulate resource monitoring - in practice would use actual system metrics
                current_time = datetime.now(timezone.utc).isoformat()
                
                for resource_type in ResourceType:
                    # Mock resource usage - in practice would get real metrics
                    usage = self._get_mock_resource_usage(sandbox, resource_type)
                    
                    resource_usage = sandbox.current_usage[resource_type]
                    resource_usage.current_usage = usage
                    resource_usage.peak_usage = max(resource_usage.peak_usage, usage)
                    resource_usage.usage_history.append(usage)
                    resource_usage.last_updated = current_time
                    
                    # Keep history limited
                    if len(resource_usage.usage_history) > 100:
                        resource_usage.usage_history.pop(0)
                    
                    # Calculate average
                    if resource_usage.usage_history:
                        resource_usage.average_usage = sum(resource_usage.usage_history) / len(resource_usage.usage_history)
                
                # Update sandbox activity
                sandbox.last_activity = current_time
                
                await asyncio.sleep(1)  # Monitor every second
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error monitoring resources for sandbox {sandbox.sandbox_id}: {e}")
                await asyncio.sleep(5)
    
    def _get_mock_resource_usage(self, sandbox: TenantSandbox, resource_type: ResourceType) -> float:
        """Get mock resource usage - in practice would use real system metrics"""
        import random
        
        # Simulate realistic usage patterns
        base_usage = {
            ResourceType.CPU: random.uniform(0.1, 2.0),
            ResourceType.MEMORY: random.uniform(0.5, 4.0),
            ResourceType.STORAGE: random.uniform(1.0, 10.0),
            ResourceType.NETWORK: random.uniform(1.0, 50.0),
            ResourceType.WORKFLOWS: len(sandbox.active_workflows),
            ResourceType.API_CALLS: random.uniform(0, 10)
        }
        
        return base_usage.get(resource_type, 0.0)
    
    async def _track_workflow_start(
        self,
        sandbox: TenantSandbox,
        workflow_id: str,
        execution_id: str
    ) -> None:
        """Track workflow start in sandbox"""
        
        sandbox.active_workflows[execution_id] = {
            'workflow_id': workflow_id,
            'execution_id': execution_id,
            'started_at': datetime.now(timezone.utc).isoformat(),
            'status': 'running'
        }
        
        # Update workflow resource usage
        workflow_usage = sandbox.current_usage[ResourceType.WORKFLOWS]
        workflow_usage.current_usage = len(sandbox.active_workflows)
    
    async def _track_workflow_completion(
        self,
        sandbox: TenantSandbox,
        workflow_id: str,
        execution_id: str,
        execution_time: float,
        success: bool
    ) -> None:
        """Track workflow completion in sandbox"""
        
        if execution_id in sandbox.active_workflows:
            sandbox.active_workflows[execution_id].update({
                'completed_at': datetime.now(timezone.utc).isoformat(),
                'execution_time_seconds': execution_time,
                'status': 'completed' if success else 'failed'
            })
            
            # Remove from active workflows after a delay (for monitoring)
            await asyncio.sleep(5)
            if execution_id in sandbox.active_workflows:
                del sandbox.active_workflows[execution_id]
        
        # Update workflow resource usage
        workflow_usage = sandbox.current_usage[ResourceType.WORKFLOWS]
        workflow_usage.current_usage = len(sandbox.active_workflows)
    
    async def get_tenant_resource_usage(self, tenant_id: int) -> Dict[str, Any]:
        """Get comprehensive resource usage for tenant"""
        
        sandbox = await self.get_tenant_sandbox(tenant_id)
        if not sandbox:
            return {'error': 'Sandbox not found'}
        
        usage_summary = {
            'tenant_id': tenant_id,
            'sandbox_id': sandbox.sandbox_id,
            'isolation_level': sandbox.isolation_level.value,
            'status': sandbox.status,
            'last_activity': sandbox.last_activity,
            'resource_usage': {},
            'active_workflows': len(sandbox.active_workflows),
            'resource_utilization': {}
        }
        
        for resource_type, usage in sandbox.current_usage.items():
            limit = sandbox.resource_limits[resource_type]
            utilization = (usage.current_usage / limit) * 100 if limit > 0 else 0
            
            usage_summary['resource_usage'][resource_type.value] = {
                'current': usage.current_usage,
                'peak': usage.peak_usage,
                'average': usage.average_usage,
                'limit': limit,
                'utilization_percent': utilization,
                'last_updated': usage.last_updated
            }
            
            usage_summary['resource_utilization'][resource_type.value] = utilization
        
        return usage_summary
    
    async def detect_noisy_neighbors(self, threshold_percent: float = 80.0) -> List[Dict[str, Any]]:
        """Detect tenants consuming excessive resources (noisy neighbors)"""
        
        noisy_neighbors = []
        
        with self._sandbox_lock:
            for tenant_id, sandbox in self.sandboxes.items():
                tenant_issues = []
                
                for resource_type, usage in sandbox.current_usage.items():
                    limit = sandbox.resource_limits[resource_type]
                    if limit > 0:
                        utilization = (usage.current_usage / limit) * 100
                        
                        if utilization >= threshold_percent:
                            tenant_issues.append({
                                'resource_type': resource_type.value,
                                'utilization_percent': utilization,
                                'current_usage': usage.current_usage,
                                'limit': limit,
                                'peak_usage': usage.peak_usage
                            })
                
                if tenant_issues:
                    noisy_neighbors.append({
                        'tenant_id': tenant_id,
                        'sandbox_id': sandbox.sandbox_id,
                        'issues': tenant_issues,
                        'total_issues': len(tenant_issues),
                        'last_activity': sandbox.last_activity
                    })
        
        # Sort by number of issues (most problematic first)
        noisy_neighbors.sort(key=lambda x: x['total_issues'], reverse=True)
        
        return noisy_neighbors
    
    async def suspend_tenant_sandbox(self, tenant_id: int, reason: str) -> bool:
        """Suspend tenant sandbox due to policy violations or resource abuse"""
        
        sandbox = await self.get_tenant_sandbox(tenant_id)
        if not sandbox:
            return False
        
        sandbox.status = "suspended"
        
        # Cancel all active workflows
        for execution_id in list(sandbox.active_workflows.keys()):
            sandbox.active_workflows[execution_id]['status'] = 'cancelled'
            sandbox.active_workflows[execution_id]['cancelled_reason'] = reason
        
        logger.warning(f"Suspended tenant sandbox {tenant_id}: {reason}")
        return True
    
    async def resume_tenant_sandbox(self, tenant_id: int) -> bool:
        """Resume suspended tenant sandbox"""
        
        sandbox = await self.get_tenant_sandbox(tenant_id)
        if not sandbox:
            return False
        
        sandbox.status = "active"
        
        logger.info(f"Resumed tenant sandbox {tenant_id}")
        return True
    
    async def get_sandbox_statistics(self) -> Dict[str, Any]:
        """Get overall sandbox statistics"""
        
        with self._sandbox_lock:
            total_sandboxes = len(self.sandboxes)
            active_sandboxes = sum(1 for s in self.sandboxes.values() if s.status == "active")
            suspended_sandboxes = sum(1 for s in self.sandboxes.values() if s.status == "suspended")
            
            total_workflows = sum(len(s.active_workflows) for s in self.sandboxes.values())
            
            # Resource utilization across all tenants
            resource_utilization = {}
            for resource_type in ResourceType:
                total_usage = sum(
                    s.current_usage[resource_type].current_usage 
                    for s in self.sandboxes.values()
                )
                total_limit = sum(
                    s.resource_limits[resource_type] 
                    for s in self.sandboxes.values()
                )
                
                utilization = (total_usage / total_limit * 100) if total_limit > 0 else 0
                resource_utilization[resource_type.value] = {
                    'total_usage': total_usage,
                    'total_limit': total_limit,
                    'utilization_percent': utilization
                }
        
        return {
            'total_sandboxes': total_sandboxes,
            'active_sandboxes': active_sandboxes,
            'suspended_sandboxes': suspended_sandboxes,
            'total_active_workflows': total_workflows,
            'resource_utilization': resource_utilization,
            'timestamp': datetime.now(timezone.utc).isoformat()
        }

# Global tenant isolation sandbox
tenant_isolation_sandbox = TenantIsolationSandbox()
