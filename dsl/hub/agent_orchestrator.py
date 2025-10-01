"""
Agent Orchestrator - Intelligent routing and execution management
Manages the execution of RBA agents with load balancing, failover, and scaling
"""

import asyncio
import json
import time
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum

from ..orchestrator import DSLOrchestrator
from ..parser import DSLWorkflow
from .workflow_registry import WorkflowRegistry, WorkflowSpec

class ExecutionPriority(Enum):
    """Execution priority levels based on SLA tiers"""
    LOW = 1      # T2 tenants
    MEDIUM = 2   # T1 tenants  
    HIGH = 3     # T0 tenants
    CRITICAL = 4 # Emergency workflows

@dataclass
class ExecutionRequest:
    """Request for workflow execution"""
    request_id: str
    workflow_id: str
    user_context: Dict[str, Any]
    input_data: Dict[str, Any] = field(default_factory=dict)
    priority: ExecutionPriority = ExecutionPriority.MEDIUM
    requested_at: datetime = field(default_factory=datetime.utcnow)
    timeout_seconds: int = 300
    retry_count: int = 0
    max_retries: int = 3

@dataclass
class ExecutionResult:
    """Result of workflow execution"""
    request_id: str
    workflow_id: str
    status: str  # 'completed', 'failed', 'timeout', 'cancelled'
    result: Dict[str, Any] = field(default_factory=dict)
    execution_time_ms: int = 0
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    error_message: Optional[str] = None
    evidence_pack_id: Optional[str] = None

class AgentOrchestrator:
    """
    Intelligent orchestrator for managing RBA agent execution
    Provides load balancing, failover, scaling, and SLA-aware routing
    """
    
    def __init__(self, registry: WorkflowRegistry, orchestrator: DSLOrchestrator):
        self.registry = registry
        self.orchestrator = orchestrator
        self.execution_queue = asyncio.PriorityQueue()
        self.active_executions = {}
        self.execution_history = []
        self.max_concurrent_executions = 10
        self.running = False
        
        # Performance tracking
        self.metrics = {
            'total_executions': 0,
            'successful_executions': 0,
            'failed_executions': 0,
            'avg_execution_time_ms': 0,
            'queue_length': 0
        }
    
    async def start(self):
        """Start the orchestrator"""
        self.running = True
        # Start the execution worker
        asyncio.create_task(self._execution_worker())
        print("🚀 Agent Orchestrator started")
    
    async def stop(self):
        """Stop the orchestrator"""
        self.running = False
        print("🛑 Agent Orchestrator stopped")
    
    async def execute_workflow(self, request: ExecutionRequest) -> ExecutionResult:
        """
        Execute a workflow with intelligent routing and management
        """
        try:
            # Validate workflow exists in registry
            workflow_spec = None
            for spec in self.registry._registry_cache.values():
                if spec.workflow_id == request.workflow_id:
                    workflow_spec = spec
                    break
            
            if not workflow_spec:
                return ExecutionResult(
                    request_id=request.request_id,
                    workflow_id=request.workflow_id,
                    status='failed',
                    error_message=f"Workflow {request.workflow_id} not found in registry"
                )
            
            # Add to execution queue with priority
            priority_score = self._calculate_priority_score(request, workflow_spec)
            await self.execution_queue.put((priority_score, request))
            
            # Wait for execution completion (in real implementation, this would be async)
            result = await self._wait_for_execution(request)
            
            # Update registry statistics
            await self.registry.update_execution_stats(
                request.workflow_id, 
                result.execution_time_ms, 
                result.status == 'completed'
            )
            
            return result
            
        except Exception as e:
            return ExecutionResult(
                request_id=request.request_id,
                workflow_id=request.workflow_id,
                status='failed',
                error_message=f"Orchestrator error: {str(e)}"
            )
    
    async def get_execution_status(self, request_id: str) -> Optional[ExecutionResult]:
        """Get status of an execution"""
        # Check active executions
        if request_id in self.active_executions:
            return self.active_executions[request_id]
        
        # Check history
        for result in self.execution_history:
            if result.request_id == request_id:
                return result
        
        return None
    
    async def get_queue_status(self) -> Dict[str, Any]:
        """Get current queue and execution status"""
        return {
            'queue_length': self.execution_queue.qsize(),
            'active_executions': len(self.active_executions),
            'max_concurrent': self.max_concurrent_executions,
            'metrics': self.metrics
        }
    
    async def cancel_execution(self, request_id: str) -> bool:
        """Cancel an execution"""
        if request_id in self.active_executions:
            result = self.active_executions[request_id]
            result.status = 'cancelled'
            result.completed_at = datetime.utcnow()
            del self.active_executions[request_id]
            self.execution_history.append(result)
            return True
        return False
    
    def _calculate_priority_score(self, request: ExecutionRequest, spec: WorkflowSpec) -> int:
        """Calculate priority score for queue ordering (lower = higher priority)"""
        base_priority = request.priority.value
        
        # Adjust based on SLA tier (from user context)
        sla_tier = request.user_context.get('sla_tier', 'T2')
        if sla_tier == 'T0':
            base_priority -= 10  # Highest priority
        elif sla_tier == 'T1':
            base_priority -= 5   # Medium priority
        
        # Adjust based on workflow success rate
        if spec.success_rate > 0.9:
            base_priority -= 2   # Reliable workflows get priority
        elif spec.success_rate < 0.7:
            base_priority += 2   # Unreliable workflows get lower priority
        
        # Add timestamp to ensure FIFO for same priority
        timestamp_factor = int(request.requested_at.timestamp()) 
        
        return base_priority * 1000 + timestamp_factor % 1000
    
    async def _execution_worker(self):
        """Background worker that processes the execution queue"""
        while self.running:
            try:
                if len(self.active_executions) < self.max_concurrent_executions:
                    # Get next request from queue (with timeout to check running status)
                    try:
                        priority_score, request = await asyncio.wait_for(
                            self.execution_queue.get(), timeout=1.0
                        )
                        # Execute in background
                        asyncio.create_task(self._execute_request(request))
                    except asyncio.TimeoutError:
                        # No requests in queue, continue loop
                        continue
                else:
                    # Too many active executions, wait a bit
                    await asyncio.sleep(0.1)
                    
            except Exception as e:
                print(f"❌ Execution worker error: {e}")
                await asyncio.sleep(1)
    
    async def _execute_request(self, request: ExecutionRequest):
        """Execute a single workflow request"""
        result = ExecutionResult(
            request_id=request.request_id,
            workflow_id=request.workflow_id,
            started_at=datetime.utcnow()
        )
        
        try:
            # Add to active executions
            self.active_executions[request.request_id] = result
            
            # Load workflow from storage
            workflow = await self.registry.storage.load_workflow(request.workflow_id)
            if not workflow:
                raise Exception(f"Workflow {request.workflow_id} not found in storage")
            
            # Execute workflow using DSL orchestrator
            start_time = time.time()
            execution_result = await self.orchestrator.execute_workflow(
                workflow=workflow,
                user_context=request.user_context,
                input_data=request.input_data
            )
            end_time = time.time()
            
            # Update result
            result.status = execution_result.get('status', 'completed')
            result.result = execution_result
            result.execution_time_ms = int((end_time - start_time) * 1000)
            result.completed_at = datetime.utcnow()
            result.evidence_pack_id = execution_result.get('evidence_pack_id')
            
            # Update metrics
            self.metrics['total_executions'] += 1
            if result.status == 'completed':
                self.metrics['successful_executions'] += 1
            else:
                self.metrics['failed_executions'] += 1
            
            # Update average execution time
            total_time = self.metrics['avg_execution_time_ms'] * (self.metrics['total_executions'] - 1)
            self.metrics['avg_execution_time_ms'] = int(
                (total_time + result.execution_time_ms) / self.metrics['total_executions']
            )
            
        except Exception as e:
            result.status = 'failed'
            result.error_message = str(e)
            result.completed_at = datetime.utcnow()
            self.metrics['total_executions'] += 1
            self.metrics['failed_executions'] += 1
            
        finally:
            # Move from active to history
            if request.request_id in self.active_executions:
                del self.active_executions[request.request_id]
            
            self.execution_history.append(result)
            
            # Keep history limited to last 1000 executions
            if len(self.execution_history) > 1000:
                self.execution_history = self.execution_history[-1000:]
    
    async def _wait_for_execution(self, request: ExecutionRequest) -> ExecutionResult:
        """Wait for execution to complete (simplified for now)"""
        # In a real implementation, this would use async events or callbacks
        timeout = request.timeout_seconds
        start_time = time.time()
        
        while time.time() - start_time < timeout:
            # Check if execution is complete
            result = await self.get_execution_status(request.request_id)
            if result and result.status in ['completed', 'failed', 'cancelled']:
                return result
            
            await asyncio.sleep(0.1)
        
        # Timeout
        await self.cancel_execution(request.request_id)
        return ExecutionResult(
            request_id=request.request_id,
            workflow_id=request.workflow_id,
            status='timeout',
            error_message=f"Execution timed out after {timeout} seconds"
        )

    async def get_performance_metrics(self) -> Dict[str, Any]:
        """Get comprehensive performance metrics"""
        return {
            'orchestrator_metrics': self.metrics,
            'queue_status': await self.get_queue_status(),
            'recent_executions': [
                {
                    'workflow_id': r.workflow_id,
                    'status': r.status,
                    'execution_time_ms': r.execution_time_ms,
                    'completed_at': r.completed_at.isoformat() if r.completed_at else None
                }
                for r in self.execution_history[-10:]  # Last 10 executions
            ]
        }
    
    async def suggest_workflow_optimizations(self) -> List[Dict[str, Any]]:
        """Suggest optimizations based on execution patterns"""
        suggestions = []
        
        # Analyze execution patterns
        if len(self.execution_history) > 10:
            # Find slow workflows
            slow_workflows = {}
            for result in self.execution_history[-100:]:
                if result.status == 'completed':
                    if result.workflow_id not in slow_workflows:
                        slow_workflows[result.workflow_id] = []
                    slow_workflows[result.workflow_id].append(result.execution_time_ms)
            
            for workflow_id, times in slow_workflows.items():
                avg_time = sum(times) / len(times)
                if avg_time > 5000:  # Slower than 5 seconds
                    suggestions.append({
                        'type': 'performance',
                        'workflow_id': workflow_id,
                        'issue': 'slow_execution',
                        'avg_time_ms': int(avg_time),
                        'recommendation': 'Consider optimizing queries or adding caching'
                    })
        
        return suggestions
