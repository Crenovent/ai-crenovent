"""
Dry-Run Executor Mode - Task 6.1.21
====================================
Execute workflows without actual ML inference for testing and validation.

Features:
- Run workflows without executing ML nodes
- Return stub results for testing
- Validate workflow structure and parameters
- Test governance policies without side effects
- Performance testing without compute costs
"""

import logging
import uuid
from typing import Dict, List, Any, Optional
from datetime import datetime
from dataclasses import dataclass, field
from enum import Enum

logger = logging.getLogger(__name__)


class DryRunMode(str, Enum):
    """Dry run execution modes"""
    STUB = "stub"  # Return predefined stub results
    VALIDATE = "validate"  # Only validate structure, no execution
    TRACE = "trace"  # Execute and trace without side effects
    PERFORMANCE = "performance"  # Test performance without actual compute


@dataclass
class DryRunConfig:
    """Configuration for dry-run execution"""
    mode: DryRunMode = DryRunMode.STUB
    stub_values: Dict[str, Any] = field(default_factory=dict)
    validate_policies: bool = True
    validate_schemas: bool = True
    capture_metrics: bool = True
    tenant_id: str = "1300"
    user_id: int = 1319


@dataclass
class DryRunResult:
    """Result from dry-run execution"""
    execution_id: str
    workflow_id: str
    mode: DryRunMode
    success: bool
    
    # Validation results
    structure_valid: bool = True
    validation_errors: List[str] = field(default_factory=list)
    validation_warnings: List[str] = field(default_factory=list)
    
    # Execution trace
    steps_executed: List[Dict[str, Any]] = field(default_factory=list)
    stub_results: Dict[str, Any] = field(default_factory=dict)
    
    # Performance metrics
    execution_time_ms: int = 0
    steps_count: int = 0
    nodes_validated: int = 0
    
    # Governance checks
    policy_checks_passed: int = 0
    policy_checks_failed: int = 0
    policy_violations: List[str] = field(default_factory=list)
    
    # Metadata
    timestamp: datetime = field(default_factory=datetime.utcnow)
    error_message: Optional[str] = None


class DryRunExecutor:
    """
    Dry-run executor for RBIA workflows
    
    Executes workflows without actual ML inference for:
    - Testing workflow structure
    - Validating parameters and policies
    - Performance testing
    - Cost estimation without spending
    """
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self._stub_generators = self._initialize_stub_generators()
    
    def _initialize_stub_generators(self) -> Dict[str, callable]:
        """Initialize stub result generators for different node types"""
        return {
            'ml_predict': self._generate_ml_predict_stub,
            'ml_score': self._generate_ml_score_stub,
            'ml_classify': self._generate_ml_classify_stub,
            'ml_explain': self._generate_ml_explain_stub,
            'filter': self._generate_filter_stub,
            'transform': self._generate_transform_stub,
            'aggregate': self._generate_aggregate_stub,
        }
    
    async def execute_dry_run(
        self,
        workflow_definition: Dict[str, Any],
        input_data: Dict[str, Any],
        config: DryRunConfig
    ) -> DryRunResult:
        """
        Execute workflow in dry-run mode
        
        Args:
            workflow_definition: Workflow DSL definition
            input_data: Input data for the workflow
            config: Dry-run configuration
            
        Returns:
            DryRunResult with validation and execution details
        """
        execution_id = str(uuid.uuid4())
        start_time = datetime.utcnow()
        
        self.logger.info(f"🧪 Starting dry-run execution: {execution_id} (mode: {config.mode})")
        
        result = DryRunResult(
            execution_id=execution_id,
            workflow_id=workflow_definition.get('id', 'unknown'),
            mode=config.mode
        )
        
        try:
            # Step 1: Validate workflow structure
            if config.mode in [DryRunMode.VALIDATE, DryRunMode.TRACE]:
                await self._validate_workflow_structure(workflow_definition, result)
            
            # Step 2: Validate input schemas
            if config.validate_schemas:
                await self._validate_input_schemas(workflow_definition, input_data, result)
            
            # Step 3: Execute workflow steps with stubs
            if config.mode in [DryRunMode.STUB, DryRunMode.TRACE]:
                await self._execute_workflow_steps(
                    workflow_definition, input_data, config, result
                )
            
            # Step 4: Validate governance policies
            if config.validate_policies:
                await self._validate_policies(workflow_definition, result)
            
            # Step 5: Calculate metrics
            execution_time = int((datetime.utcnow() - start_time).total_seconds() * 1000)
            result.execution_time_ms = execution_time
            result.success = len(result.validation_errors) == 0
            
            self.logger.info(
                f"✅ Dry-run completed: {execution_id} "
                f"({result.steps_count} steps, {execution_time}ms)"
            )
            
        except Exception as e:
            result.success = False
            result.error_message = str(e)
            result.validation_errors.append(f"Execution error: {str(e)}")
            self.logger.error(f"❌ Dry-run failed: {execution_id} - {e}")
        
        return result
    
    async def _validate_workflow_structure(
        self,
        workflow_def: Dict[str, Any],
        result: DryRunResult
    ):
        """Validate workflow structure and dependencies"""
        errors = []
        warnings = []
        
        # Check required fields
        if 'id' not in workflow_def:
            errors.append("Workflow missing required 'id' field")
        
        if 'steps' not in workflow_def and 'nodes' not in workflow_def:
            errors.append("Workflow missing 'steps' or 'nodes' field")
        
        # Validate steps
        steps = workflow_def.get('steps', workflow_def.get('nodes', []))
        result.steps_count = len(steps)
        
        for idx, step in enumerate(steps):
            step_id = step.get('id', f'step_{idx}')
            
            # Check step structure
            if 'type' not in step:
                errors.append(f"Step '{step_id}' missing 'type' field")
            
            # Check dependencies
            if 'depends_on' in step:
                deps = step['depends_on']
                if not isinstance(deps, list):
                    deps = [deps]
                
                for dep in deps:
                    dep_exists = any(s.get('id') == dep for s in steps)
                    if not dep_exists:
                        errors.append(f"Step '{step_id}' depends on non-existent step '{dep}'")
            
            result.nodes_validated += 1
        
        result.validation_errors.extend(errors)
        result.validation_warnings.extend(warnings)
        result.structure_valid = len(errors) == 0
    
    async def _validate_input_schemas(
        self,
        workflow_def: Dict[str, Any],
        input_data: Dict[str, Any],
        result: DryRunResult
    ):
        """Validate input data against workflow schema"""
        expected_inputs = workflow_def.get('inputs', {})
        
        for input_name, input_spec in expected_inputs.items():
            if input_spec.get('required', False) and input_name not in input_data:
                result.validation_errors.append(
                    f"Required input '{input_name}' not provided"
                )
            
            if input_name in input_data:
                # Validate type if specified
                expected_type = input_spec.get('type')
                if expected_type:
                    actual_value = input_data[input_name]
                    if not self._validate_type(actual_value, expected_type):
                        result.validation_warnings.append(
                            f"Input '{input_name}' type mismatch: "
                            f"expected {expected_type}, got {type(actual_value).__name__}"
                        )
    
    async def _execute_workflow_steps(
        self,
        workflow_def: Dict[str, Any],
        input_data: Dict[str, Any],
        config: DryRunConfig,
        result: DryRunResult
    ):
        """Execute workflow steps with stub results"""
        steps = workflow_def.get('steps', workflow_def.get('nodes', []))
        execution_context = input_data.copy()
        
        for idx, step in enumerate(steps):
            step_id = step.get('id', f'step_{idx}')
            step_type = step.get('type', 'unknown')
            
            self.logger.debug(f"🔄 Executing dry-run step: {step_id} ({step_type})")
            
            # Generate stub result
            stub_result = await self._generate_stub_result(
                step_type, step, execution_context, config
            )
            
            # Store step execution details
            step_execution = {
                'step_id': step_id,
                'step_type': step_type,
                'stub_result': stub_result,
                'execution_order': idx + 1,
                'timestamp': datetime.utcnow().isoformat()
            }
            result.steps_executed.append(step_execution)
            
            # Update execution context
            if step.get('output_key'):
                execution_context[step['output_key']] = stub_result
            
            result.stub_results[step_id] = stub_result
    
    async def _generate_stub_result(
        self,
        step_type: str,
        step_config: Dict[str, Any],
        context: Dict[str, Any],
        config: DryRunConfig
    ) -> Any:
        """Generate stub result for a step"""
        
        # Check if custom stub provided
        step_id = step_config.get('id', 'unknown')
        if step_id in config.stub_values:
            return config.stub_values[step_id]
        
        # Use type-specific stub generator
        generator = self._stub_generators.get(step_type, self._generate_default_stub)
        return await generator(step_config, context)
    
    async def _generate_ml_predict_stub(
        self,
        step_config: Dict[str, Any],
        context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Generate stub for ML prediction"""
        return {
            'prediction': 0.75,
            'confidence': 0.85,
            'model_id': step_config.get('model_id', 'stub_model'),
            'model_version': '1.0.0',
            'explanation': 'Stub prediction result',
            'feature_importance': {
                'feature_1': 0.3,
                'feature_2': 0.2,
                'feature_3': 0.15
            },
            'is_stub': True
        }
    
    async def _generate_ml_score_stub(
        self,
        step_config: Dict[str, Any],
        context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Generate stub for ML scoring"""
        score_range = step_config.get('score_range', [0, 100])
        return {
            'score': (score_range[0] + score_range[1]) / 2,
            'percentile': 75.0,
            'category': 'medium',
            'model_id': step_config.get('model_id', 'stub_scorer'),
            'is_stub': True
        }
    
    async def _generate_ml_classify_stub(
        self,
        step_config: Dict[str, Any],
        context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Generate stub for ML classification"""
        classes = step_config.get('classes', ['class_a', 'class_b'])
        return {
            'predicted_class': classes[0],
            'probabilities': {cls: 1.0 / len(classes) for cls in classes},
            'confidence': 0.8,
            'is_stub': True
        }
    
    async def _generate_ml_explain_stub(
        self,
        step_config: Dict[str, Any],
        context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Generate stub for ML explanation"""
        return {
            'explanation_type': step_config.get('method', 'shap'),
            'feature_attributions': {
                'feature_1': 0.25,
                'feature_2': 0.20,
                'feature_3': 0.15
            },
            'narrative': 'Stub explanation for testing',
            'is_stub': True
        }
    
    async def _generate_filter_stub(
        self,
        step_config: Dict[str, Any],
        context: Dict[str, Any]
    ) -> List[Any]:
        """Generate stub for filter operation"""
        input_data = context.get(step_config.get('input', 'data'), [])
        # Return half the input for stub
        return input_data[:len(input_data) // 2] if input_data else []
    
    async def _generate_transform_stub(
        self,
        step_config: Dict[str, Any],
        context: Dict[str, Any]
    ) -> Any:
        """Generate stub for transform operation"""
        return {'transformed': True, 'is_stub': True}
    
    async def _generate_aggregate_stub(
        self,
        step_config: Dict[str, Any],
        context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Generate stub for aggregate operation"""
        return {
            'count': 100,
            'sum': 1000.0,
            'average': 10.0,
            'is_stub': True
        }
    
    async def _generate_default_stub(
        self,
        step_config: Dict[str, Any],
        context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Generate default stub for unknown step types"""
        return {
            'result': 'stub_value',
            'step_type': step_config.get('type', 'unknown'),
            'is_stub': True
        }
    
    async def _validate_policies(
        self,
        workflow_def: Dict[str, Any],
        result: DryRunResult
    ):
        """Validate governance policies"""
        policies = workflow_def.get('policies', [])
        
        for policy in policies:
            policy_id = policy.get('id', 'unknown')
            
            # Check policy structure
            if 'rules' not in policy:
                result.policy_violations.append(
                    f"Policy '{policy_id}' missing 'rules' field"
                )
                result.policy_checks_failed += 1
            else:
                result.policy_checks_passed += 1
    
    def _validate_type(self, value: Any, expected_type: str) -> bool:
        """Validate value against expected type"""
        type_map = {
            'string': str,
            'number': (int, float),
            'integer': int,
            'boolean': bool,
            'array': list,
            'object': dict
        }
        expected = type_map.get(expected_type.lower())
        if expected:
            return isinstance(value, expected)
        return True


# API helper functions
async def execute_workflow_dry_run(
    workflow_definition: Dict[str, Any],
    input_data: Dict[str, Any],
    mode: str = "stub",
    stub_values: Optional[Dict[str, Any]] = None,
    tenant_id: str = "1300",
    user_id: int = 1319
) -> DryRunResult:
    """
    Execute a workflow in dry-run mode
    
    Args:
        workflow_definition: Workflow DSL definition
        input_data: Input data for the workflow
        mode: Dry-run mode (stub, validate, trace, performance)
        stub_values: Custom stub values for specific steps
        tenant_id: Tenant ID for multi-tenant context
        user_id: User ID for audit trail
        
    Returns:
        DryRunResult with execution details
    """
    config = DryRunConfig(
        mode=DryRunMode(mode),
        stub_values=stub_values or {},
        tenant_id=tenant_id,
        user_id=user_id
    )
    
    executor = DryRunExecutor()
    return await executor.execute_dry_run(workflow_definition, input_data, config)


if __name__ == "__main__":
    # Example usage
    import asyncio
    
    async def test_dry_run():
        workflow = {
            'id': 'test_workflow',
            'name': 'Test Workflow',
            'inputs': {
                'customer_data': {'type': 'object', 'required': True}
            },
            'steps': [
                {
                    'id': 'predict_churn',
                    'type': 'ml_predict',
                    'model_id': 'churn_model_v2',
                    'output_key': 'churn_prediction'
                },
                {
                    'id': 'score_risk',
                    'type': 'ml_score',
                    'model_id': 'risk_scorer',
                    'depends_on': ['predict_churn'],
                    'score_range': [0, 100]
                }
            ]
        }
        
        input_data = {
            'customer_data': {'customer_id': '123', 'mrr': 5000}
        }
        
        result = await execute_workflow_dry_run(workflow, input_data)
        
        print(f"✅ Dry-run completed: {result.execution_id}")
        print(f"   Success: {result.success}")
        print(f"   Steps executed: {result.steps_count}")
        print(f"   Execution time: {result.execution_time_ms}ms")
        print(f"   Validation errors: {len(result.validation_errors)}")
        
        if result.validation_errors:
            print(f"   Errors: {result.validation_errors}")
        
        print(f"\n   Stub results:")
        for step_id, stub_result in result.stub_results.items():
            print(f"     {step_id}: {stub_result}")
    
    asyncio.run(test_dry_run())

