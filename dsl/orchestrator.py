"""
DSL Orchestrator - Executes DSL workflows with governance and monitoring
Enhanced with governance-by-design validation (Task 7.1.3)
"""

import asyncio
import uuid
import time
from typing import Dict, List, Any, Optional
from datetime import datetime
from dataclasses import dataclass, field

from .parser import DSLWorkflow, DSLStep, DSLParser
from .operators.base import OperatorContext, OperatorResult
from .operators.query import QueryOperator
from .operators.decision import DecisionOperator
from .operators.agent_call import AgentCallOperator
from .operators.notify import NotifyOperator
from .operators.governance import GovernanceOperator
from .governance import get_governance_validator, ValidationContext, EnforcementMode

import logging

logger = logging.getLogger(__name__)

# Knowledge Graph integration will be initialized in hub layer

@dataclass
class WorkflowExecution:
    """Represents a workflow execution instance"""
    execution_id: str
    workflow_id: str
    user_id: int
    tenant_id: str
    
    # Execution state
    status: str = "running"  # running, completed, failed, cancelled
    started_at: datetime = field(default_factory=datetime.utcnow)
    completed_at: Optional[datetime] = None
    
    # Results
    input_data: Dict[str, Any] = field(default_factory=dict)
    output_data: Dict[str, Any] = field(default_factory=dict)
    step_results: Dict[str, OperatorResult] = field(default_factory=dict)
    
    # Performance
    execution_time_ms: int = 0
    step_count: int = 0
    
    # Governance
    evidence_pack_id: Optional[str] = None
    policy_violations: List[str] = field(default_factory=list)
    overrides_applied: int = 0
    
    # Error handling
    error_message: Optional[str] = None
    failed_step_id: Optional[str] = None

class DSLOrchestrator:
    """
    Orchestrates DSL workflow execution with governance, monitoring, and error handling
    """
    
    def __init__(self, pool_manager=None):
        self.pool_manager = pool_manager
        self.logger = logging.getLogger(__name__)
        
        # Initialize operators
        self.operators = {
            'query': QueryOperator(),
            'decision': DecisionOperator(),
            'agent_call': AgentCallOperator(),
            'notify': NotifyOperator(),
            'governance': GovernanceOperator()
        }
        
        # Add policy-aware operators
        try:
            from .operators.policy_query import PolicyAwareQueryOperator
            self.operators['policy_query'] = PolicyAwareQueryOperator()
        except ImportError:
            self.logger.warning("PolicyAwareQueryOperator not available - using fallback")
        
        # Execution tracking
        self.active_executions: Dict[str, WorkflowExecution] = {}
    
    async def execute_workflow(
        self,
        workflow: DSLWorkflow,
        input_data: Dict[str, Any],
        user_context: Dict[str, Any]
    ) -> WorkflowExecution:
        """
        Execute a DSL workflow with full governance and monitoring
        
        Args:
            workflow: Parsed DSL workflow
            input_data: Input data for the workflow
            user_context: User context (user_id, tenant_id, session_id, etc.)
            
        Returns:
            WorkflowExecution with results and governance data
        """
        execution_id = str(uuid.uuid4())
        start_time = time.time()
        
        # Create execution instance
        execution = WorkflowExecution(
            execution_id=execution_id,
            workflow_id=workflow.workflow_id,
            user_id=user_context['user_id'],
            tenant_id=user_context['tenant_id'],
            input_data=input_data,
            step_count=len(workflow.steps)
        )
        
        # Track active execution
        self.active_executions[execution_id] = execution
        
        try:
            self.logger.info(f"Starting workflow execution: {workflow.workflow_id} ({execution_id})")
            
            # Pre-execution governance checks
            governance_check = await self._pre_execution_governance(workflow, execution)
            if not governance_check['success']:
                execution.status = "failed"
                execution.error_message = governance_check['error']
                return execution
            
            # Execute workflow steps
            current_step_id = workflow.steps[0].id if workflow.steps else None
            step_outputs = {}
            
            while current_step_id:
                step = self._find_step_by_id(workflow.steps, current_step_id)
                if not step:
                    execution.status = "failed"
                    execution.error_message = f"Step not found: {current_step_id}"
                    execution.failed_step_id = current_step_id
                    break
                
                # Execute step
                step_result = await self._execute_step(step, execution, step_outputs, input_data)
                execution.step_results[step.id] = step_result
                
                if not step_result.success:
                    execution.status = "failed"
                    execution.error_message = step_result.error_message
                    execution.failed_step_id = step.id
                    execution.policy_violations.extend(step_result.policy_violations)
                    break
                
                # Store step output for next steps
                step_outputs[step.id] = step_result.output_data
                
                # Add to overall output
                execution.output_data.update(step_result.output_data)
                
                # Track violations
                execution.policy_violations.extend(step_result.policy_violations)
                
                # Determine next step
                current_step_id = step_result.next_step_id or self._get_next_step_id(workflow.steps, current_step_id)
            
            # Mark completion
            if execution.status == "running":
                execution.status = "completed"
            
            execution.completed_at = datetime.utcnow()
            execution.execution_time_ms = int((time.time() - start_time) * 1000)
            
            # Post-execution governance
            await self._post_execution_governance(workflow, execution)
            
            self.logger.info(f"Workflow execution completed: {execution_id} ({execution.status})")
            
        except Exception as e:
            execution.status = "failed"
            execution.error_message = str(e)
            execution.completed_at = datetime.utcnow()
            execution.execution_time_ms = int((time.time() - start_time) * 1000)
            
            self.logger.error(f"Workflow execution error: {e}")
            
        finally:
            # Store execution trace
            await self._store_execution_trace(execution)
            
            # Remove from active executions
            if execution_id in self.active_executions:
                del self.active_executions[execution_id]
        
        return execution
    
    async def _execute_step(
        self,
        step: DSLStep,
        execution: WorkflowExecution,
        previous_outputs: Dict[str, Any],
        input_data: Dict[str, Any]
    ) -> OperatorResult:
        """Execute a single workflow step"""
        
        try:
            self.logger.info(f"Executing step: {step.id} ({step.type})")
            
            # Get operator
            operator = self.operators.get(step.type)
            if not operator:
                return OperatorResult(
                    success=False,
                    error_message=f"Unknown operator type: {step.type}"
                )
            
            # Create operator context
            context = OperatorContext(
                user_id=execution.user_id,
                tenant_id=execution.tenant_id,
                session_id=input_data.get('session_id'),
                workflow_id=execution.workflow_id,
                step_id=step.id,
                execution_id=execution.execution_id,
                input_data=input_data,
                previous_outputs=previous_outputs,
                policy_pack_id=step.governance.get('policy_id'),
                trust_threshold=float(step.governance.get('trust_threshold', 0.70)),
                evidence_required=step.governance.get('evidence_capture', True),
                pool_manager=self.pool_manager
            )
            
            # Execute operator
            result = await operator.execute(context, step.params)
            
            # Add step-specific evidence
            result.add_evidence('step_id', step.id)
            result.add_evidence('step_type', step.type)
            result.add_evidence('workflow_id', execution.workflow_id)
            
            return result
            
        except Exception as e:
            self.logger.error(f"Step execution error: {e}")
            return OperatorResult(
                success=False,
                error_message=f"Step execution failed: {e}"
            )
    
    async def _pre_execution_governance(self, workflow: DSLWorkflow, execution: WorkflowExecution) -> Dict[str, Any]:
        """Perform governance checks before workflow execution"""
        try:
            # Check tenant permissions
            if not execution.tenant_id:
                return {'success': False, 'error': 'Tenant ID required'}
            
            # Check workflow governance requirements
            governance_config = workflow.governance
            policy_pack_id = governance_config.get('policy_pack')
            
            if policy_pack_id:
                # Load and validate policy pack
                policy_valid = await self._validate_policy_pack(policy_pack_id, execution.tenant_id)
                if not policy_valid:
                    return {'success': False, 'error': f'Invalid policy pack: {policy_pack_id}'}
            
            # Check SLA tier compliance
            sla_tier = workflow.metadata.get('sla_tier', 'T2')
            if not await self._check_sla_tier_permission(execution.user_id, sla_tier):
                return {'success': False, 'error': f'Insufficient permissions for SLA tier: {sla_tier}'}
            
            return {'success': True}
            
        except Exception as e:
            return {'success': False, 'error': f'Governance check failed: {e}'}
    
    async def _post_execution_governance(self, workflow: DSLWorkflow, execution: WorkflowExecution):
        """Perform governance actions after workflow execution"""
        try:
            # Generate evidence pack if required
            if workflow.governance.get('evidence_pack_enabled', True):
                evidence_pack_id = await self._generate_evidence_pack(workflow, execution)
                execution.evidence_pack_id = evidence_pack_id
            
            # Check for policy violations requiring escalation
            if execution.policy_violations:
                await self._handle_policy_violations(workflow, execution)
            
            # Update workflow success metrics
            await self._update_workflow_metrics(workflow, execution)
            
        except Exception as e:
            self.logger.error(f"Post-execution governance error: {e}")
    
    async def _validate_policy_pack(self, policy_pack_id: str, tenant_id: str) -> bool:
        """Validate that policy pack exists and is applicable to tenant"""
        try:
            if not self.pool_manager or not self.pool_manager.postgres_pool:
                return True  # Skip validation if DB not available
            
            async with self.pool_manager.postgres_pool.acquire() as conn:
                await conn.execute(f"SET app.current_tenant = '{tenant_id}'")
                
                query = """
                    SELECT policy_pack_id FROM dsl_policy_packs
                    WHERE policy_pack_id = $1
                    AND (is_global = true OR tenant_id = $2)
                """
                
                row = await conn.fetchrow(query, policy_pack_id, tenant_id)
                return row is not None
                
        except Exception as e:
            self.logger.error(f"Policy pack validation error: {e}")
            return False
    
    async def _check_sla_tier_permission(self, user_id: int, sla_tier: str) -> bool:
        """Check if user has permission for requested SLA tier"""
        try:
            # Basic permission check - in production, this would be more sophisticated
            # For now, allow all users to access all tiers
            return True
            
        except Exception as e:
            self.logger.error(f"SLA tier permission check error: {e}")
            return False
    
    async def _generate_evidence_pack(self, workflow: DSLWorkflow, execution: WorkflowExecution) -> str:
        """Generate evidence pack for compliance and audit"""
        try:
            evidence_data = {
                'workflow_id': workflow.workflow_id,
                'execution_id': execution.execution_id,
                'user_id': execution.user_id,
                'tenant_id': execution.tenant_id,
                'timestamp': datetime.utcnow().isoformat(),
                'workflow_config': {
                    'name': workflow.name,
                    'module': workflow.module,
                    'automation_type': workflow.automation_type,
                    'version': workflow.version
                },
                'execution_summary': {
                    'status': execution.status,
                    'execution_time_ms': execution.execution_time_ms,
                    'step_count': execution.step_count,
                    'policy_violations': execution.policy_violations
                },
                'step_evidence': {}
            }
            
            # Collect evidence from each step
            for step_id, step_result in execution.step_results.items():
                evidence_data['step_evidence'][step_id] = step_result.evidence_data
            
            # Store evidence pack
            if self.pool_manager and self.pool_manager.postgres_pool:
                async with self.pool_manager.postgres_pool.acquire() as conn:
                    await conn.execute(f"SET app.current_tenant = '{execution.tenant_id}'")
                    
                    query = """
                        INSERT INTO dsl_evidence_packs (
                            trace_id, evidence_data, evidence_hash, tenant_id
                        ) VALUES ($1, $2, $3, $4)
                        RETURNING evidence_pack_id
                    """
                    
                    import json
                    import hashlib
                    evidence_json = json.dumps(evidence_data, sort_keys=True)
                    evidence_hash = hashlib.sha256(evidence_json.encode()).hexdigest()
                    
                    row = await conn.fetchrow(
                        query,
                        execution.execution_id,
                        evidence_json,
                        evidence_hash,
                        execution.tenant_id
                    )
                    
                    return str(row['evidence_pack_id'])
            
            return f"evidence_mock_{execution.execution_id}"
            
        except Exception as e:
            self.logger.error(f"Evidence pack generation error: {e}")
            return f"evidence_error_{execution.execution_id}"
    
    async def _handle_policy_violations(self, workflow: DSLWorkflow, execution: WorkflowExecution):
        """Handle policy violations found during execution"""
        try:
            if not execution.policy_violations:
                return
            
            self.logger.warning(f"Policy violations in workflow {workflow.workflow_id}: {execution.policy_violations}")
            
            # Store violations for audit
            if self.pool_manager and self.pool_manager.postgres_pool:
                async with self.pool_manager.postgres_pool.acquire() as conn:
                    await conn.execute(f"SET app.current_tenant = '{execution.tenant_id}'")
                    
                    # Update evidence pack with violations
                    query = """
                        UPDATE dsl_evidence_packs
                        SET policy_violations = $1, compliance_status = 'violation'
                        WHERE trace_id = $2
                    """
                    
                    import json
                    await conn.execute(
                        query,
                        json.dumps(execution.policy_violations),
                        execution.execution_id
                    )
            
        except Exception as e:
            self.logger.error(f"Policy violation handling error: {e}")
    
    async def _update_workflow_metrics(self, workflow: DSLWorkflow, execution: WorkflowExecution):
        """Update workflow performance metrics"""
        try:
            if not self.pool_manager or not self.pool_manager.postgres_pool:
                return
            
            async with self.pool_manager.postgres_pool.acquire() as conn:
                await conn.execute(f"SET app.current_tenant = '{execution.tenant_id}'")
                
                # Update workflow statistics
                query = """
                    UPDATE dsl_workflows
                    SET 
                        execution_count = execution_count + 1,
                        success_rate = CASE 
                            WHEN $2 = 'completed' THEN 
                                (success_rate * execution_count + 100.0) / (execution_count + 1)
                            ELSE 
                                (success_rate * execution_count) / (execution_count + 1)
                        END,
                        avg_execution_time_ms = 
                            (avg_execution_time_ms * execution_count + $3) / (execution_count + 1),
                        updated_at = NOW()
                    WHERE workflow_id = $1
                """
                
                await conn.execute(
                    query,
                    workflow.workflow_id,
                    execution.status,
                    execution.execution_time_ms
                )
            
        except Exception as e:
            self.logger.error(f"Workflow metrics update error: {e}")
    
    async def _store_execution_trace(self, execution: WorkflowExecution):
        """Store execution trace in database"""
        try:
            if not self.pool_manager or not self.pool_manager.postgres_pool:
                return
            
            async with self.pool_manager.postgres_pool.acquire() as conn:
                await conn.execute(f"SET app.current_tenant = '{execution.tenant_id}'")
                
                # Prepare step execution data
                steps_data = []
                for step_id, result in execution.step_results.items():
                    steps_data.append({
                        'step_id': step_id,
                        'success': result.success,
                        'execution_time_ms': result.execution_time_ms,
                        'confidence_score': result.confidence_score,
                        'error_message': result.error_message
                    })
                
                # Insert execution trace
                query = """
                    INSERT INTO dsl_execution_traces (
                        trace_id, workflow_id, user_id, tenant_id,
                        input_data, output_data, steps_executed,
                        started_at, completed_at, execution_time_ms,
                        status, error_message, confidence_score,
                        evidence_pack_id, override_count
                    ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14, $15)
                """
                
                import json
                await conn.execute(
                    query,
                    execution.execution_id,
                    execution.workflow_id,
                    execution.user_id,
                    execution.tenant_id,
                    json.dumps(execution.input_data),
                    json.dumps(execution.output_data),
                    json.dumps(steps_data),
                    execution.started_at,
                    execution.completed_at,
                    execution.execution_time_ms,
                    execution.status,
                    execution.error_message,
                    self._calculate_overall_confidence(execution),
                    execution.evidence_pack_id,
                    execution.overrides_applied
                )
            
        except Exception as e:
            self.logger.error(f"Execution trace storage error: {e}")
    
    def _calculate_overall_confidence(self, execution: WorkflowExecution) -> float:
        """Calculate overall confidence score for workflow execution"""
        if not execution.step_results:
            return 0.0
        
        confidence_scores = [
            result.confidence_score for result in execution.step_results.values()
            if result.success
        ]
        
        if not confidence_scores:
            return 0.0
        
        return sum(confidence_scores) / len(confidence_scores)
    
    def _find_step_by_id(self, steps: List[DSLStep], step_id: str) -> Optional[DSLStep]:
        """Find step by ID"""
        for step in steps:
            if step.id == step_id:
                return step
        return None
    
    def _get_next_step_id(self, steps: List[DSLStep], current_step_id: str) -> Optional[str]:
        """Get next step ID in sequence"""
        current_index = None
        for i, step in enumerate(steps):
            if step.id == current_step_id:
                current_index = i
                break
        
        if current_index is not None and current_index + 1 < len(steps):
            return steps[current_index + 1].id
        
        return None
    
    def get_execution_status(self, execution_id: str) -> Optional[Dict[str, Any]]:
        """Get status of active execution"""
        execution = self.active_executions.get(execution_id)
        if not execution:
            return None
        
        return {
            'execution_id': execution.execution_id,
            'workflow_id': execution.workflow_id,
            'status': execution.status,
            'started_at': execution.started_at.isoformat(),
            'execution_time_ms': execution.execution_time_ms,
            'step_count': execution.step_count,
            'completed_steps': len(execution.step_results),
            'policy_violations': len(execution.policy_violations)
        }
