"""
Complete RBA Workflow Execution Engine
Core engine for executing Rule-Based Automation workflows with full governance and compliance
"""

from typing import Dict, List, Any, Optional, Union, Callable, AsyncGenerator
from datetime import datetime, timedelta
from enum import Enum
from pydantic import BaseModel, Field
from abc import ABC, abstractmethod
import json
import logging
import asyncio
import uuid
from contextlib import asynccontextmanager
import traceback

from ..saas.dsl_templates import SaaSDSLTemplateManager, SaaSWorkflowType
from ..saas.parameter_packs import SaaSParameterPackManager
from ..saas.schemas import SaaSSchemaFactory, SaaSWorkflowTrace, SaaSAuditTrail
from ..saas.compiler_enhancements import SaaSRuntimeEnhancer, PolicyInjector

logger = logging.getLogger(__name__)

class ExecutionStatus(Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    PAUSED = "paused"

class StepType(Enum):
    INITIALIZATION = "initialization"
    DATA_QUERY = "data_query"
    CALCULATION = "calculation"
    RULE_EVALUATION = "rule_evaluation"
    NOTIFICATION = "notification"
    TASK_CREATION = "task_creation"
    REPORT_GENERATION = "report_generation"
    AUDIT_LOGGING = "audit_logging"
    ML_SCORING = "ml_scoring"
    EXTERNAL_API = "external_api"
    DATA_UPDATE = "data_update"
    HUMAN_APPROVAL = "human_approval"

class ExecutionContext(BaseModel):
    """Execution context for workflow runs"""
    execution_id: str
    workflow_id: str
    tenant_id: str
    user_id: Optional[str] = None
    started_at: datetime
    parameters: Dict[str, Any] = {}
    variables: Dict[str, Any] = {}
    step_results: Dict[str, Any] = {}
    traces: List[SaaSWorkflowTrace] = []
    errors: List[Dict[str, Any]] = []
    warnings: List[Dict[str, Any]] = []
    
    class Config:
        arbitrary_types_allowed = True

class StepExecutionResult(BaseModel):
    """Result of a single step execution"""
    step_id: str
    step_type: str
    status: ExecutionStatus
    started_at: datetime
    completed_at: Optional[datetime] = None
    duration_ms: int = 0
    output_data: Dict[str, Any] = {}
    error: Optional[str] = None
    warnings: List[str] = []
    trace_id: Optional[str] = None

class WorkflowExecutionResult(BaseModel):
    """Result of complete workflow execution"""
    execution_id: str
    workflow_id: str
    status: ExecutionStatus
    started_at: datetime
    completed_at: Optional[datetime] = None
    duration_ms: int = 0
    step_results: List[StepExecutionResult] = []
    final_output: Dict[str, Any] = {}
    error: Optional[str] = None
    audit_trail_id: Optional[str] = None
    compliance_status: Dict[str, str] = {}

# Abstract base classes for step executors
class BaseStepExecutor(ABC):
    """Base class for step executors"""
    
    def __init__(self):
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
    
    @abstractmethod
    async def execute(
        self,
        step_definition: Dict[str, Any],
        context: ExecutionContext
    ) -> StepExecutionResult:
        """Execute a workflow step"""
        pass
    
    def _create_result(
        self,
        step_definition: Dict[str, Any],
        status: ExecutionStatus,
        started_at: datetime,
        output_data: Dict[str, Any] = None,
        error: str = None,
        warnings: List[str] = None
    ) -> StepExecutionResult:
        """Helper to create step execution result"""
        completed_at = datetime.utcnow()
        duration_ms = int((completed_at - started_at).total_seconds() * 1000)
        
        return StepExecutionResult(
            step_id=step_definition["id"],
            step_type=step_definition["type"],
            status=status,
            started_at=started_at,
            completed_at=completed_at,
            duration_ms=duration_ms,
            output_data=output_data or {},
            error=error,
            warnings=warnings or []
        )

# Concrete step executors
class InitializationExecutor(BaseStepExecutor):
    """Executor for initialization steps"""
    
    async def execute(
        self,
        step_definition: Dict[str, Any],
        context: ExecutionContext
    ) -> StepExecutionResult:
        started_at = datetime.utcnow()
        
        try:
            config = step_definition.get("config", {})
            
            # Validate data sources
            if config.get("validate_data_sources"):
                # Mock validation - in real implementation, check database connections
                await asyncio.sleep(0.1)
            
            # Check permissions
            if config.get("check_permissions"):
                # Mock permission check
                if not context.user_id:
                    return self._create_result(
                        step_definition, ExecutionStatus.FAILED, started_at,
                        error="User ID required for permission check"
                    )
            
            # Log execution start
            if config.get("log_execution_start"):
                self.logger.info(f"Starting workflow execution: {context.execution_id}")
            
            output_data = {
                "initialization_completed": True,
                "validation_passed": True,
                "permissions_verified": bool(context.user_id)
            }
            
            return self._create_result(
                step_definition, ExecutionStatus.COMPLETED, started_at, output_data
            )
            
        except Exception as e:
            self.logger.error(f"Initialization step failed: {e}")
            return self._create_result(
                step_definition, ExecutionStatus.FAILED, started_at, error=str(e)
            )

class DataQueryExecutor(BaseStepExecutor):
    """Executor for data query steps"""
    
    async def execute(
        self,
        step_definition: Dict[str, Any],
        context: ExecutionContext
    ) -> StepExecutionResult:
        started_at = datetime.utcnow()
        
        try:
            config = step_definition.get("config", {})
            query = config.get("query", "")
            parameters = config.get("parameters", {})
            max_results = config.get("max_results", 1000)
            
            # Replace parameter placeholders
            for param_name, param_value in parameters.items():
                if isinstance(param_value, str) and param_value.startswith("{{") and param_value.endswith("}}"):
                    # Extract parameter reference
                    param_ref = param_value[2:-2].strip()
                    if param_ref.startswith("parameters."):
                        param_key = param_ref.split(".", 1)[1]
                        actual_value = context.parameters.get(param_key)
                        parameters[param_name] = actual_value
            
            # Mock query execution - in real implementation, execute against database
            await asyncio.sleep(0.2)  # Simulate query time
            
            # Generate mock results based on query type
            if "stalled" in query.lower() and "opportunities" in query.lower():
                mock_results = self._generate_mock_stalled_opportunities(parameters)
            elif "pipeline" in query.lower() and "metrics" in query.lower():
                mock_results = self._generate_mock_pipeline_metrics()
            else:
                mock_results = []
            
            # Apply max_results limit
            if len(mock_results) > max_results:
                mock_results = mock_results[:max_results]
                warnings = [f"Results limited to {max_results} records"]
            else:
                warnings = []
            
            output_data = {
                "query_executed": query,
                "parameters_used": parameters,
                "results": mock_results,
                "result_count": len(mock_results)
            }
            
            # Store results in context variables
            output_variable = step_definition.get("output", {}).get("variable")
            if output_variable:
                context.variables[output_variable] = mock_results
            
            return self._create_result(
                step_definition, ExecutionStatus.COMPLETED, started_at, 
                output_data, warnings=warnings
            )
            
        except Exception as e:
            self.logger.error(f"Data query step failed: {e}")
            return self._create_result(
                step_definition, ExecutionStatus.FAILED, started_at, error=str(e)
            )
    
    def _generate_mock_stalled_opportunities(self, parameters: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate mock stalled opportunities data"""
        threshold_days = parameters.get("stalled_opportunity_threshold_days", 15)
        
        return [
            {
                "opportunity_id": f"opp_{i:03d}",
                "opportunity_name": f"Enterprise Deal {i}",
                "current_stage": "negotiation",
                "stage_entry_date": (datetime.utcnow() - timedelta(days=threshold_days + i)).isoformat(),
                "amount": 50000 + (i * 5000),
                "close_date": (datetime.utcnow() + timedelta(days=30 - i)).isoformat(),
                "owner_id": f"user_{i % 5}",
                "days_in_stage": threshold_days + i,
                "account_name": f"Acme Corp {i}",
                "owner_name": f"Sales Rep {i % 5}",
                "owner_email": f"rep{i % 5}@company.com"
            }
            for i in range(1, 8)  # Generate 7 stalled opportunities
        ]
    
    def _generate_mock_pipeline_metrics(self) -> List[Dict[str, Any]]:
        """Generate mock pipeline metrics data"""
        return [
            {
                "metric_name": "total_pipeline_value",
                "value": 2500000,
                "currency": "USD"
            },
            {
                "metric_name": "quarterly_quota",
                "value": 1000000,
                "currency": "USD"
            }
        ]

class CalculationExecutor(BaseStepExecutor):
    """Executor for calculation steps"""
    
    async def execute(
        self,
        step_definition: Dict[str, Any],
        context: ExecutionContext
    ) -> StepExecutionResult:
        started_at = datetime.utcnow()
        
        try:
            config = step_definition.get("config", {})
            calculations = config.get("calculations", [])
            
            results = {}
            
            for calc in calculations:
                calc_name = calc["name"]
                formula = calc["formula"]
                
                # Mock calculation execution
                result_value = await self._execute_calculation(formula, context)
                results[calc_name] = result_value
            
            output_data = {
                "calculations_completed": len(calculations),
                "results": results
            }
            
            # Store results in context variables
            output_variable = step_definition.get("output", {}).get("variable")
            if output_variable:
                context.variables[output_variable] = results
            
            return self._create_result(
                step_definition, ExecutionStatus.COMPLETED, started_at, output_data
            )
            
        except Exception as e:
            self.logger.error(f"Calculation step failed: {e}")
            return self._create_result(
                step_definition, ExecutionStatus.FAILED, started_at, error=str(e)
            )
    
    async def _execute_calculation(self, formula: str, context: ExecutionContext) -> float:
        """Execute a calculation formula"""
        # Mock calculation based on formula
        if "total_pipeline_value" in formula:
            return 2500000.0
        elif "quarterly_quota" in formula:
            return 1000000.0
        elif "pipeline_coverage_ratio" in formula:
            return 2.5
        elif "stalled_count" in formula:
            stalled_opps = context.variables.get("stalled_opportunities", [])
            return float(len(stalled_opps))
        elif "hygiene_score" in formula:
            return 0.85
        else:
            return 100.0

class RuleEvaluationExecutor(BaseStepExecutor):
    """Executor for rule evaluation steps"""
    
    async def execute(
        self,
        step_definition: Dict[str, Any],
        context: ExecutionContext
    ) -> StepExecutionResult:
        started_at = datetime.utcnow()
        
        try:
            config = step_definition.get("config", {})
            rules = config.get("rules", [])
            
            evaluation_results = []
            
            for rule in rules:
                rule_result = await self._evaluate_rule(rule, context)
                evaluation_results.append(rule_result)
            
            output_data = {
                "rules_evaluated": len(rules),
                "evaluation_results": evaluation_results,
                "all_rules_passed": all(r["passed"] for r in evaluation_results),
                "failed_rules": [r for r in evaluation_results if not r["passed"]]
            }
            
            # Store results in context variables
            output_variable = step_definition.get("output", {}).get("variable")
            if output_variable:
                context.variables[output_variable] = output_data
            
            return self._create_result(
                step_definition, ExecutionStatus.COMPLETED, started_at, output_data
            )
            
        except Exception as e:
            self.logger.error(f"Rule evaluation step failed: {e}")
            return self._create_result(
                step_definition, ExecutionStatus.FAILED, started_at, error=str(e)
            )
    
    async def _evaluate_rule(self, rule: Dict[str, Any], context: ExecutionContext) -> Dict[str, Any]:
        """Evaluate a single rule"""
        rule_id = rule.get("id", "unknown")
        rule_name = rule.get("name", "Unknown Rule")
        condition = rule.get("condition", "")
        
        # Mock rule evaluation - in real implementation, parse and evaluate condition
        if "pipeline_coverage_ratio" in condition and ">=" in condition:
            threshold = float(condition.split(">=")[1].strip())
            actual_value = context.variables.get("pipeline_metrics", {}).get("pipeline_coverage_ratio", 0)
            passed = actual_value >= threshold
        elif "stalled_count" in condition and "<=" in condition:
            threshold = float(condition.split("<=")[1].strip())
            actual_value = context.variables.get("pipeline_metrics", {}).get("stalled_count", 0)
            passed = actual_value <= threshold
        elif "hygiene_score" in condition and ">=" in condition:
            threshold = float(condition.split(">=")[1].strip())
            actual_value = context.variables.get("pipeline_metrics", {}).get("hygiene_score", 0)
            passed = actual_value >= threshold
        else:
            passed = True  # Default to passing
        
        return {
            "rule_id": rule_id,
            "rule_name": rule_name,
            "condition": condition,
            "passed": passed,
            "severity": rule.get("severity", "medium"),
            "action_required": rule.get("action_required", False) and not passed
        }

class NotificationExecutor(BaseStepExecutor):
    """Executor for notification steps"""
    
    async def execute(
        self,
        step_definition: Dict[str, Any],
        context: ExecutionContext
    ) -> StepExecutionResult:
        started_at = datetime.utcnow()
        
        try:
            config = step_definition.get("config", {})
            notification_rules = config.get("notification_rules", [])
            
            notifications_sent = []
            
            for rule in notification_rules:
                if await self._should_send_notification(rule, context):
                    notification_result = await self._send_notification(rule, context)
                    notifications_sent.append(notification_result)
            
            output_data = {
                "notifications_sent": len(notifications_sent),
                "notification_details": notifications_sent
            }
            
            return self._create_result(
                step_definition, ExecutionStatus.COMPLETED, started_at, output_data
            )
            
        except Exception as e:
            self.logger.error(f"Notification step failed: {e}")
            return self._create_result(
                step_definition, ExecutionStatus.FAILED, started_at, error=str(e)
            )
    
    async def _should_send_notification(self, rule: Dict[str, Any], context: ExecutionContext) -> bool:
        """Check if notification should be sent based on condition"""
        condition = rule.get("condition", "")
        
        # Mock condition evaluation
        if "coverage_ratio_check.failed == true" in condition:
            rule_results = context.variables.get("rule_evaluation_results", {})
            failed_rules = rule_results.get("failed_rules", [])
            return any(r["rule_id"] == "coverage_ratio_check" for r in failed_rules)
        
        return True  # Default to sending
    
    async def _send_notification(self, rule: Dict[str, Any], context: ExecutionContext) -> Dict[str, Any]:
        """Send a notification"""
        # Mock notification sending
        await asyncio.sleep(0.1)
        
        return {
            "template": rule.get("template", "default"),
            "recipients": rule.get("recipients", []),
            "channels": rule.get("channels", ["email"]),
            "priority": rule.get("priority", "medium"),
            "sent_at": datetime.utcnow().isoformat(),
            "status": "sent"
        }

# Main workflow engine
class RBAWorkflowEngine:
    """Complete RBA Workflow Execution Engine"""
    
    def __init__(self):
        self.template_manager = SaaSDSLTemplateManager()
        self.parameter_manager = SaaSParameterPackManager()
        self.schema_factory = SaaSSchemaFactory()
        self.runtime_enhancer = SaaSRuntimeEnhancer()
        self.policy_injector = PolicyInjector()
        
        # Step executors registry
        self.step_executors = {
            StepType.INITIALIZATION: InitializationExecutor(),
            StepType.DATA_QUERY: DataQueryExecutor(),
            StepType.CALCULATION: CalculationExecutor(),
            StepType.RULE_EVALUATION: RuleEvaluationExecutor(),
            StepType.NOTIFICATION: NotificationExecutor(),
            # Add more executors as needed
        }
        
        self.active_executions: Dict[str, ExecutionContext] = {}
        self.logger = logging.getLogger(__name__)
    
    async def execute_workflow(
        self,
        workflow_type: SaaSWorkflowType,
        tenant_id: str,
        user_id: Optional[str] = None,
        parameters: Dict[str, Any] = None,
        execution_mode: str = "sync"
    ) -> WorkflowExecutionResult:
        """Execute a complete workflow"""
        
        execution_id = str(uuid.uuid4())
        started_at = datetime.utcnow()
        
        try:
            # Get workflow template
            template = self.template_manager.get_template(workflow_type)
            workflow_id = template["workflow_id"]
            
            # Merge parameters with defaults
            parameter_pack_id = f"saas_{workflow_type.value}_v1.0"
            merged_parameters = self.parameter_manager.merge_parameters(
                parameter_pack_id, parameters or {}
            )
            
            # Inject policies
            enhanced_template = self.policy_injector.inject_policies_into_workflow(template)
            
            # Create execution context
            context = ExecutionContext(
                execution_id=execution_id,
                workflow_id=workflow_id,
                tenant_id=tenant_id,
                user_id=user_id,
                started_at=started_at,
                parameters=merged_parameters
            )
            
            self.active_executions[execution_id] = context
            
            self.logger.info(f"Starting workflow execution: {execution_id} ({workflow_id})")
            
            # Execute workflow steps
            step_results = []
            
            async with self.runtime_enhancer.enhanced_execution_context(
                execution_id, workflow_id, tenant_id, user_id
            ) as runtime_context:
                
                for step_definition in enhanced_template.get("steps", []):
                    # Check dependencies
                    if not await self._check_step_dependencies(step_definition, step_results):
                        step_result = StepExecutionResult(
                            step_id=step_definition["id"],
                            step_type=step_definition["type"],
                            status=ExecutionStatus.FAILED,
                            started_at=datetime.utcnow(),
                            error="Step dependencies not satisfied"
                        )
                        step_results.append(step_result)
                        break
                    
                    # Execute step
                    step_result = await self._execute_step(step_definition, context)
                    step_results.append(step_result)
                    
                    # Store step result in context
                    context.step_results[step_definition["id"]] = step_result
                    
                    # Check if step failed
                    if step_result.status == ExecutionStatus.FAILED:
                        self.logger.error(f"Step failed: {step_definition['id']} - {step_result.error}")
                        break
            
            # Calculate final status
            completed_at = datetime.utcnow()
            duration_ms = int((completed_at - started_at).total_seconds() * 1000)
            
            if any(step.status == ExecutionStatus.FAILED for step in step_results):
                final_status = ExecutionStatus.FAILED
                final_output = {"error": "One or more steps failed"}
            else:
                final_status = ExecutionStatus.COMPLETED
                final_output = self._generate_final_output(context, step_results)
            
            # Create audit trail
            audit_trail_id = await self._create_audit_trail(context, step_results)
            
            # Create final result
            result = WorkflowExecutionResult(
                execution_id=execution_id,
                workflow_id=workflow_id,
                status=final_status,
                started_at=started_at,
                completed_at=completed_at,
                duration_ms=duration_ms,
                step_results=step_results,
                final_output=final_output,
                audit_trail_id=audit_trail_id,
                compliance_status={"SOX": "compliant", "GDPR": "compliant"}  # Mock
            )
            
            self.logger.info(f"Workflow execution completed: {execution_id} - {final_status.value}")
            
            return result
            
        except Exception as e:
            self.logger.error(f"Workflow execution failed: {execution_id} - {e}")
            self.logger.error(traceback.format_exc())
            
            completed_at = datetime.utcnow()
            duration_ms = int((completed_at - started_at).total_seconds() * 1000)
            
            return WorkflowExecutionResult(
                execution_id=execution_id,
                workflow_id=template.get("workflow_id", "unknown"),
                status=ExecutionStatus.FAILED,
                started_at=started_at,
                completed_at=completed_at,
                duration_ms=duration_ms,
                error=str(e)
            )
        
        finally:
            # Cleanup
            self.active_executions.pop(execution_id, None)
    
    async def _execute_step(
        self,
        step_definition: Dict[str, Any],
        context: ExecutionContext
    ) -> StepExecutionResult:
        """Execute a single workflow step"""
        
        step_type_str = step_definition.get("type", "unknown")
        step_id = step_definition.get("id", "unknown")
        
        try:
            # Get step type enum
            step_type = StepType(step_type_str)
        except ValueError:
            self.logger.warning(f"Unknown step type: {step_type_str}")
            step_type = StepType.INITIALIZATION  # Default fallback
        
        # Get executor for step type
        executor = self.step_executors.get(step_type)
        if not executor:
            self.logger.error(f"No executor found for step type: {step_type}")
            return StepExecutionResult(
                step_id=step_id,
                step_type=step_type_str,
                status=ExecutionStatus.FAILED,
                started_at=datetime.utcnow(),
                error=f"No executor available for step type: {step_type_str}"
            )
        
        # Execute step with policy enforcement
        try:
            if hasattr(self.runtime_enhancer, 'execute_step_with_policies'):
                # Use enhanced execution with policies
                async def step_executor_wrapper(step_def, exec_context):
                    return await executor.execute(step_def, context)
                
                enhanced_result = await self.runtime_enhancer.execute_step_with_policies(
                    step_definition, context.dict(), step_executor_wrapper
                )
                
                if enhanced_result.get("success", False):
                    return enhanced_result["step_result"]
                else:
                    return StepExecutionResult(
                        step_id=step_id,
                        step_type=step_type_str,
                        status=ExecutionStatus.FAILED,
                        started_at=datetime.utcnow(),
                        error=enhanced_result.get("error", "Step execution failed")
                    )
            else:
                # Direct execution without policy enforcement
                return await executor.execute(step_definition, context)
                
        except Exception as e:
            self.logger.error(f"Step execution error: {step_id} - {e}")
            return StepExecutionResult(
                step_id=step_id,
                step_type=step_type_str,
                status=ExecutionStatus.FAILED,
                started_at=datetime.utcnow(),
                error=str(e)
            )
    
    async def _check_step_dependencies(
        self,
        step_definition: Dict[str, Any],
        completed_steps: List[StepExecutionResult]
    ) -> bool:
        """Check if step dependencies are satisfied"""
        
        depends_on = step_definition.get("depends_on", [])
        if not depends_on:
            return True  # No dependencies
        
        completed_step_ids = {step.step_id for step in completed_steps if step.status == ExecutionStatus.COMPLETED}
        
        for dependency in depends_on:
            if dependency not in completed_step_ids:
                return False
        
        return True
    
    def _generate_final_output(
        self,
        context: ExecutionContext,
        step_results: List[StepExecutionResult]
    ) -> Dict[str, Any]:
        """Generate final workflow output"""
        
        return {
            "workflow_completed": True,
            "steps_executed": len(step_results),
            "successful_steps": len([s for s in step_results if s.status == ExecutionStatus.COMPLETED]),
            "failed_steps": len([s for s in step_results if s.status == ExecutionStatus.FAILED]),
            "total_duration_ms": sum(s.duration_ms for s in step_results),
            "variables": context.variables,
            "execution_summary": {
                "pipeline_metrics": context.variables.get("pipeline_metrics", {}),
                "rule_evaluation_results": context.variables.get("rule_evaluation_results", {}),
                "notifications_sent": len([s for s in step_results if s.step_type == "notification"])
            }
        }
    
    async def _create_audit_trail(
        self,
        context: ExecutionContext,
        step_results: List[StepExecutionResult]
    ) -> str:
        """Create audit trail for workflow execution"""
        
        try:
            audit_trail = self.schema_factory.create_audit_trail(
                tenant_id=context.tenant_id,
                workflow_type="pipeline_hygiene",  # Should be determined dynamically
                workflow_execution_id=context.execution_id,
                audit_type="execution",
                execution_summary=context.dict(),
                steps_executed=[step.dict() for step in step_results],
                total_execution_time_ms=sum(s.duration_ms for s in step_results)
            )
            
            # In real implementation, save to database
            self.logger.info(f"Created audit trail: {audit_trail.audit_id}")
            
            return audit_trail.audit_id
            
        except Exception as e:
            self.logger.error(f"Failed to create audit trail: {e}")
            return f"audit_error_{context.execution_id}"
    
    def get_execution_status(self, execution_id: str) -> Optional[ExecutionContext]:
        """Get current execution status"""
        return self.active_executions.get(execution_id)
    
    def list_active_executions(self) -> List[str]:
        """List all active execution IDs"""
        return list(self.active_executions.keys())
    
    async def cancel_execution(self, execution_id: str) -> bool:
        """Cancel a running execution"""
        if execution_id in self.active_executions:
            # In real implementation, implement proper cancellation
            self.active_executions.pop(execution_id)
            self.logger.info(f"Cancelled execution: {execution_id}")
            return True
        return False

# Example usage and testing
if __name__ == "__main__":
    import asyncio
    
    async def test_workflow_execution():
        """Test complete workflow execution"""
        
        engine = RBAWorkflowEngine()
        
        # Test pipeline hygiene workflow
        result = await engine.execute_workflow(
            workflow_type=SaaSWorkflowType.PIPELINE_HYGIENE,
            tenant_id="tenant_123",
            user_id="user_456",
            parameters={
                "stalled_opportunity_threshold_days": 20,
                "pipeline_coverage_minimum": 2.5
            }
        )
        
        print(f"Execution Result:")
        print(f"  ID: {result.execution_id}")
        print(f"  Status: {result.status.value}")
        print(f"  Duration: {result.duration_ms}ms")
        print(f"  Steps: {len(result.step_results)}")
        print(f"  Success Rate: {len([s for s in result.step_results if s.status == ExecutionStatus.COMPLETED])/len(result.step_results)*100:.1f}%")
        
        if result.error:
            print(f"  Error: {result.error}")
        else:
            print(f"  Final Output Keys: {list(result.final_output.keys())}")
    
    # Run test
    asyncio.run(test_workflow_execution())
