"""
DSL Reconciliation Engine
========================

Task 7.1.16: Configure DSL reconciliation checks
Validates DSL workflow definitions against execution traces to ensure consistency
"""

import asyncio
import json
import logging
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime, timedelta
import hashlib

logger = logging.getLogger(__name__)

class ReconciliationStatus(Enum):
    """Reconciliation check status"""
    PASSED = "passed"
    FAILED = "failed"
    WARNING = "warning"
    SKIPPED = "skipped"

class ReconciliationSeverity(Enum):
    """Severity levels for reconciliation issues"""
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    INFO = "info"

@dataclass
class ReconciliationIssue:
    """Individual reconciliation issue"""
    issue_id: str
    severity: ReconciliationSeverity
    category: str
    message: str
    dsl_reference: Optional[str] = None
    trace_reference: Optional[str] = None
    expected_value: Optional[Any] = None
    actual_value: Optional[Any] = None
    recommendation: Optional[str] = None

@dataclass
class ReconciliationResult:
    """Result of DSL reconciliation check"""
    workflow_id: str
    execution_id: str
    status: ReconciliationStatus
    issues: List[ReconciliationIssue] = field(default_factory=list)
    checks_performed: int = 0
    checks_passed: int = 0
    checks_failed: int = 0
    checks_skipped: int = 0
    reconciliation_timestamp: datetime = field(default_factory=datetime.utcnow)
    
    def add_issue(self, issue: ReconciliationIssue):
        """Add a reconciliation issue"""
        self.issues.append(issue)
        if issue.severity in [ReconciliationSeverity.CRITICAL, ReconciliationSeverity.HIGH]:
            self.checks_failed += 1
        elif issue.severity == ReconciliationSeverity.MEDIUM:
            self.checks_failed += 1
        else:
            self.checks_passed += 1

class DSLReconciliationEngine:
    """
    DSL Reconciliation Engine - Task 7.1.16
    
    Validates consistency between DSL workflow definitions and execution traces
    """
    
    def __init__(self, pool_manager=None):
        self.pool_manager = pool_manager
        self.logger = logging.getLogger(__name__)
        
        # Reconciliation rules configuration
        self.reconciliation_rules = self._load_reconciliation_rules()
        
        # Cache for workflow definitions
        self.workflow_cache = {}
        
    def _load_reconciliation_rules(self) -> Dict[str, Any]:
        """Load reconciliation rules configuration"""
        return {
            'step_consistency': {
                'enabled': True,
                'check_step_count': True,
                'check_step_types': True,
                'check_step_sequence': True,
                'tolerance_threshold': 0.05  # 5% tolerance for minor variations
            },
            'execution_flow': {
                'enabled': True,
                'check_branching_logic': True,
                'check_conditional_paths': True,
                'check_loop_iterations': True
            },
            'data_consistency': {
                'enabled': True,
                'check_input_parameters': True,
                'check_output_schemas': True,
                'check_data_transformations': True
            },
            'governance_compliance': {
                'enabled': True,
                'check_policy_enforcement': True,
                'check_evidence_generation': True,
                'check_override_logging': True,
                'check_trust_score_calculation': True
            },
            'performance_validation': {
                'enabled': True,
                'check_execution_time': True,
                'check_resource_usage': True,
                'check_sla_compliance': True
            },
            'error_handling': {
                'enabled': True,
                'check_retry_logic': True,
                'check_fallback_paths': True,
                'check_error_propagation': True
            }
        }
    
    async def reconcile_workflow_execution(
        self,
        workflow_id: str,
        execution_id: str,
        tenant_id: int,
        deep_validation: bool = False
    ) -> ReconciliationResult:
        """
        Reconcile a specific workflow execution against its DSL definition
        
        Args:
            workflow_id: Workflow identifier
            execution_id: Execution identifier
            tenant_id: Tenant identifier
            deep_validation: Enable comprehensive validation checks
            
        Returns:
            ReconciliationResult: Detailed reconciliation results
        """
        
        result = ReconciliationResult(
            workflow_id=workflow_id,
            execution_id=execution_id,
            status=ReconciliationStatus.PASSED
        )
        
        try:
            self.logger.info(f"🔍 Starting DSL reconciliation: {workflow_id} (execution: {execution_id})")
            
            # 1. Load workflow definition and execution trace
            workflow_def, execution_trace = await self._load_workflow_and_trace(
                workflow_id, execution_id, tenant_id
            )
            
            if not workflow_def or not execution_trace:
                result.add_issue(ReconciliationIssue(
                    issue_id="DATA_MISSING",
                    severity=ReconciliationSeverity.CRITICAL,
                    category="data_availability",
                    message="Workflow definition or execution trace not found",
                    recommendation="Ensure workflow and trace data are properly stored"
                ))
                result.status = ReconciliationStatus.FAILED
                return result
            
            # 2. Step consistency validation
            if self.reconciliation_rules['step_consistency']['enabled']:
                await self._validate_step_consistency(workflow_def, execution_trace, result)
            
            # 3. Execution flow validation
            if self.reconciliation_rules['execution_flow']['enabled']:
                await self._validate_execution_flow(workflow_def, execution_trace, result)
            
            # 4. Data consistency validation
            if self.reconciliation_rules['data_consistency']['enabled']:
                await self._validate_data_consistency(workflow_def, execution_trace, result)
            
            # 5. Governance compliance validation
            if self.reconciliation_rules['governance_compliance']['enabled']:
                await self._validate_governance_compliance(workflow_def, execution_trace, result)
            
            # 6. Performance validation
            if self.reconciliation_rules['performance_validation']['enabled']:
                await self._validate_performance_metrics(workflow_def, execution_trace, result)
            
            # 7. Error handling validation
            if self.reconciliation_rules['error_handling']['enabled']:
                await self._validate_error_handling(workflow_def, execution_trace, result)
            
            # 8. Deep validation (optional)
            if deep_validation:
                await self._perform_deep_validation(workflow_def, execution_trace, result)
            
            # Determine overall status
            critical_issues = [i for i in result.issues if i.severity == ReconciliationSeverity.CRITICAL]
            high_issues = [i for i in result.issues if i.severity == ReconciliationSeverity.HIGH]
            
            if critical_issues:
                result.status = ReconciliationStatus.FAILED
            elif high_issues:
                result.status = ReconciliationStatus.WARNING
            else:
                result.status = ReconciliationStatus.PASSED
            
            self.logger.info(f"✅ DSL reconciliation completed: {result.status.value} "
                           f"({len(result.issues)} issues found)")
            
            return result
            
        except Exception as e:
            self.logger.error(f"❌ DSL reconciliation failed: {e}")
            result.add_issue(ReconciliationIssue(
                issue_id="RECONCILIATION_ERROR",
                severity=ReconciliationSeverity.CRITICAL,
                category="system_error",
                message=f"Reconciliation engine error: {str(e)}",
                recommendation="Check system logs and retry reconciliation"
            ))
            result.status = ReconciliationStatus.FAILED
            return result
    
    async def _load_workflow_and_trace(
        self,
        workflow_id: str,
        execution_id: str,
        tenant_id: int
    ) -> Tuple[Optional[Dict[str, Any]], Optional[Dict[str, Any]]]:
        """Load workflow definition and execution trace from database"""
        
        if not self.pool_manager:
            return None, None
        
        try:
            async with self.pool_manager.get_connection() as conn:
                # Load workflow definition
                workflow_query = """
                    SELECT workflow_definition, plan_hash, governance_metadata
                    FROM dsl_workflows 
                    WHERE workflow_id = $1 AND tenant_id = $2
                """
                workflow_row = await conn.fetchrow(workflow_query, workflow_id, tenant_id)
                
                # Load execution trace
                trace_query = """
                    SELECT trace_data, plan_hash, execution_status, governance_metadata,
                           execution_time_ms, trust_score, override_count
                    FROM dsl_execution_traces 
                    WHERE execution_id = $1 AND tenant_id = $2
                """
                trace_row = await conn.fetchrow(trace_query, execution_id, tenant_id)
                
                workflow_def = dict(workflow_row) if workflow_row else None
                execution_trace = dict(trace_row) if trace_row else None
                
                return workflow_def, execution_trace
                
        except Exception as e:
            self.logger.error(f"Failed to load workflow and trace data: {e}")
            return None, None
    
    async def _validate_step_consistency(
        self,
        workflow_def: Dict[str, Any],
        execution_trace: Dict[str, Any],
        result: ReconciliationResult
    ):
        """Validate step consistency between DSL and execution"""
        
        try:
            # Extract steps from workflow definition
            dsl_steps = workflow_def.get('workflow_definition', {}).get('steps', [])
            
            # Extract steps from execution trace
            trace_data = execution_trace.get('trace_data', {})
            executed_steps = trace_data.get('steps', [])
            
            result.checks_performed += 1
            
            # Check step count
            if len(dsl_steps) != len(executed_steps):
                result.add_issue(ReconciliationIssue(
                    issue_id="STEP_COUNT_MISMATCH",
                    severity=ReconciliationSeverity.HIGH,
                    category="step_consistency",
                    message=f"Step count mismatch: DSL has {len(dsl_steps)} steps, execution has {len(executed_steps)} steps",
                    expected_value=len(dsl_steps),
                    actual_value=len(executed_steps),
                    recommendation="Verify workflow execution completed all defined steps"
                ))
                return
            
            # Check step types and sequence
            for i, (dsl_step, exec_step) in enumerate(zip(dsl_steps, executed_steps)):
                dsl_step_id = dsl_step.get('id', f'step_{i}')
                exec_step_id = exec_step.get('step_id', f'step_{i}')
                
                # Check step ID consistency
                if dsl_step_id != exec_step_id:
                    result.add_issue(ReconciliationIssue(
                        issue_id=f"STEP_ID_MISMATCH_{i}",
                        severity=ReconciliationSeverity.MEDIUM,
                        category="step_consistency",
                        message=f"Step {i} ID mismatch: DSL='{dsl_step_id}', Execution='{exec_step_id}'",
                        dsl_reference=dsl_step_id,
                        trace_reference=exec_step_id,
                        recommendation="Ensure step IDs are consistent between DSL and execution"
                    ))
                
                # Check step type consistency
                dsl_step_type = dsl_step.get('type')
                exec_step_type = exec_step.get('type')
                
                if dsl_step_type != exec_step_type:
                    result.add_issue(ReconciliationIssue(
                        issue_id=f"STEP_TYPE_MISMATCH_{i}",
                        severity=ReconciliationSeverity.HIGH,
                        category="step_consistency",
                        message=f"Step {dsl_step_id} type mismatch: DSL='{dsl_step_type}', Execution='{exec_step_type}'",
                        expected_value=dsl_step_type,
                        actual_value=exec_step_type,
                        recommendation="Verify step type mapping in execution engine"
                    ))
            
            result.checks_passed += 1
            
        except Exception as e:
            result.add_issue(ReconciliationIssue(
                issue_id="STEP_VALIDATION_ERROR",
                severity=ReconciliationSeverity.MEDIUM,
                category="validation_error",
                message=f"Step consistency validation failed: {str(e)}",
                recommendation="Check step data format and structure"
            ))
    
    async def _validate_execution_flow(
        self,
        workflow_def: Dict[str, Any],
        execution_trace: Dict[str, Any],
        result: ReconciliationResult
    ):
        """Validate execution flow matches DSL definition"""
        
        try:
            result.checks_performed += 1
            
            # Extract flow information
            dsl_steps = workflow_def.get('workflow_definition', {}).get('steps', [])
            trace_data = execution_trace.get('trace_data', {})
            executed_steps = trace_data.get('steps', [])
            
            # Build expected flow from DSL
            expected_flow = self._build_expected_flow(dsl_steps)
            
            # Build actual flow from execution
            actual_flow = self._build_actual_flow(executed_steps)
            
            # Compare flows
            flow_differences = self._compare_execution_flows(expected_flow, actual_flow)
            
            for diff in flow_differences:
                result.add_issue(ReconciliationIssue(
                    issue_id=f"FLOW_MISMATCH_{diff['step_id']}",
                    severity=ReconciliationSeverity.MEDIUM,
                    category="execution_flow",
                    message=diff['message'],
                    expected_value=diff.get('expected'),
                    actual_value=diff.get('actual'),
                    recommendation="Verify conditional logic and branching in workflow execution"
                ))
            
            if not flow_differences:
                result.checks_passed += 1
            
        except Exception as e:
            result.add_issue(ReconciliationIssue(
                issue_id="FLOW_VALIDATION_ERROR",
                severity=ReconciliationSeverity.MEDIUM,
                category="validation_error",
                message=f"Execution flow validation failed: {str(e)}",
                recommendation="Check execution flow data structure"
            ))
    
    async def _validate_governance_compliance(
        self,
        workflow_def: Dict[str, Any],
        execution_trace: Dict[str, Any],
        result: ReconciliationResult
    ):
        """Validate governance compliance between DSL and execution"""
        
        try:
            result.checks_performed += 1
            
            # Check plan hash consistency
            dsl_plan_hash = workflow_def.get('plan_hash')
            exec_plan_hash = execution_trace.get('plan_hash')
            
            if dsl_plan_hash and exec_plan_hash and dsl_plan_hash != exec_plan_hash:
                result.add_issue(ReconciliationIssue(
                    issue_id="PLAN_HASH_MISMATCH",
                    severity=ReconciliationSeverity.CRITICAL,
                    category="governance_compliance",
                    message="Plan hash mismatch indicates workflow definition changed during execution",
                    expected_value=dsl_plan_hash,
                    actual_value=exec_plan_hash,
                    recommendation="Ensure workflow definition is immutable during execution"
                ))
                return
            
            # Check governance metadata consistency
            dsl_governance = workflow_def.get('governance_metadata', {})
            exec_governance = execution_trace.get('governance_metadata', {})
            
            # Validate policy pack enforcement
            dsl_policy_pack = dsl_governance.get('policy_pack_id')
            exec_policy_pack = exec_governance.get('policy_pack_id')
            
            if dsl_policy_pack and dsl_policy_pack != exec_policy_pack:
                result.add_issue(ReconciliationIssue(
                    issue_id="POLICY_PACK_MISMATCH",
                    severity=ReconciliationSeverity.HIGH,
                    category="governance_compliance",
                    message="Policy pack mismatch between DSL and execution",
                    expected_value=dsl_policy_pack,
                    actual_value=exec_policy_pack,
                    recommendation="Verify policy pack enforcement in execution engine"
                ))
            
            # Check trust score validation
            expected_trust_threshold = dsl_governance.get('trust_threshold', 0.8)
            actual_trust_score = execution_trace.get('trust_score', 1.0)
            
            if actual_trust_score < expected_trust_threshold:
                result.add_issue(ReconciliationIssue(
                    issue_id="TRUST_SCORE_VIOLATION",
                    severity=ReconciliationSeverity.HIGH,
                    category="governance_compliance",
                    message=f"Trust score {actual_trust_score} below threshold {expected_trust_threshold}",
                    expected_value=expected_trust_threshold,
                    actual_value=actual_trust_score,
                    recommendation="Review trust score calculation and workflow execution quality"
                ))
            
            result.checks_passed += 1
            
        except Exception as e:
            result.add_issue(ReconciliationIssue(
                issue_id="GOVERNANCE_VALIDATION_ERROR",
                severity=ReconciliationSeverity.MEDIUM,
                category="validation_error",
                message=f"Governance compliance validation failed: {str(e)}",
                recommendation="Check governance metadata structure"
            ))
    
    async def _validate_data_consistency(
        self,
        workflow_def: Dict[str, Any],
        execution_trace: Dict[str, Any],
        result: ReconciliationResult
    ):
        """Validate data consistency between DSL and execution"""
        
        try:
            result.checks_performed += 1
            
            # This is a placeholder for data consistency validation
            # In a real implementation, you would:
            # 1. Compare input/output schemas
            # 2. Validate data transformations
            # 3. Check parameter passing between steps
            # 4. Verify data type consistency
            
            result.checks_passed += 1
            
        except Exception as e:
            result.add_issue(ReconciliationIssue(
                issue_id="DATA_VALIDATION_ERROR",
                severity=ReconciliationSeverity.MEDIUM,
                category="validation_error",
                message=f"Data consistency validation failed: {str(e)}",
                recommendation="Check data structure and schema consistency"
            ))
    
    async def _validate_performance_metrics(
        self,
        workflow_def: Dict[str, Any],
        execution_trace: Dict[str, Any],
        result: ReconciliationResult
    ):
        """Validate performance metrics against SLA requirements"""
        
        try:
            result.checks_performed += 1
            
            # Check execution time against SLA
            governance = workflow_def.get('governance_metadata', {})
            sla_tier = governance.get('sla_tier', 'T2')
            
            # Define SLA thresholds (in milliseconds)
            sla_thresholds = {
                'T0': 5000,   # 5 seconds
                'T1': 15000,  # 15 seconds
                'T2': 60000,  # 1 minute
                'T3': 300000  # 5 minutes
            }
            
            max_execution_time = sla_thresholds.get(sla_tier, 60000)
            actual_execution_time = execution_trace.get('execution_time_ms', 0)
            
            if actual_execution_time > max_execution_time:
                result.add_issue(ReconciliationIssue(
                    issue_id="SLA_VIOLATION",
                    severity=ReconciliationSeverity.HIGH,
                    category="performance_validation",
                    message=f"Execution time {actual_execution_time}ms exceeds SLA threshold {max_execution_time}ms for tier {sla_tier}",
                    expected_value=max_execution_time,
                    actual_value=actual_execution_time,
                    recommendation=f"Optimize workflow performance or adjust SLA tier"
                ))
            
            result.checks_passed += 1
            
        except Exception as e:
            result.add_issue(ReconciliationIssue(
                issue_id="PERFORMANCE_VALIDATION_ERROR",
                severity=ReconciliationSeverity.MEDIUM,
                category="validation_error",
                message=f"Performance validation failed: {str(e)}",
                recommendation="Check performance metrics data"
            ))
    
    async def _validate_error_handling(
        self,
        workflow_def: Dict[str, Any],
        execution_trace: Dict[str, Any],
        result: ReconciliationResult
    ):
        """Validate error handling and retry logic"""
        
        try:
            result.checks_performed += 1
            
            # Check if errors were handled according to DSL definition
            trace_data = execution_trace.get('trace_data', {})
            execution_status = execution_trace.get('execution_status', 'completed')
            
            # If execution failed, check if proper error handling was applied
            if execution_status == 'failed':
                error_message = trace_data.get('error_message')
                if not error_message:
                    result.add_issue(ReconciliationIssue(
                        issue_id="MISSING_ERROR_INFO",
                        severity=ReconciliationSeverity.MEDIUM,
                        category="error_handling",
                        message="Execution failed but no error message recorded",
                        recommendation="Ensure proper error logging in execution engine"
                    ))
            
            # Check retry logic if applicable
            override_count = execution_trace.get('override_count', 0)
            if override_count > 0:
                # Validate that overrides were properly logged
                result.add_issue(ReconciliationIssue(
                    issue_id="OVERRIDES_DETECTED",
                    severity=ReconciliationSeverity.INFO,
                    category="error_handling",
                    message=f"Execution had {override_count} overrides",
                    actual_value=override_count,
                    recommendation="Review override reasons and consider workflow improvements"
                ))
            
            result.checks_passed += 1
            
        except Exception as e:
            result.add_issue(ReconciliationIssue(
                issue_id="ERROR_HANDLING_VALIDATION_ERROR",
                severity=ReconciliationSeverity.MEDIUM,
                category="validation_error",
                message=f"Error handling validation failed: {str(e)}",
                recommendation="Check error handling data structure"
            ))
    
    async def _perform_deep_validation(
        self,
        workflow_def: Dict[str, Any],
        execution_trace: Dict[str, Any],
        result: ReconciliationResult
    ):
        """Perform deep validation checks"""
        
        try:
            result.checks_performed += 1
            
            # Deep validation would include:
            # 1. Semantic analysis of workflow logic
            # 2. Cross-reference with external systems
            # 3. Historical execution pattern analysis
            # 4. Advanced anomaly detection
            
            # For now, this is a placeholder
            result.checks_passed += 1
            
        except Exception as e:
            result.add_issue(ReconciliationIssue(
                issue_id="DEEP_VALIDATION_ERROR",
                severity=ReconciliationSeverity.LOW,
                category="validation_error",
                message=f"Deep validation failed: {str(e)}",
                recommendation="Check deep validation configuration"
            ))
    
    def _build_expected_flow(self, dsl_steps: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Build expected execution flow from DSL steps"""
        # Simplified flow building - in reality this would be more complex
        flow = {}
        for i, step in enumerate(dsl_steps):
            step_id = step.get('id', f'step_{i}')
            next_steps = step.get('next_steps', [])
            flow[step_id] = {
                'type': step.get('type'),
                'next_steps': next_steps,
                'conditions': step.get('conditions', {})
            }
        return flow
    
    def _build_actual_flow(self, executed_steps: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Build actual execution flow from trace steps"""
        # Simplified flow building - in reality this would be more complex
        flow = {}
        for step in executed_steps:
            step_id = step.get('step_id')
            if step_id:
                flow[step_id] = {
                    'status': step.get('status'),
                    'execution_order': step.get('execution_order', 0)
                }
        return flow
    
    def _compare_execution_flows(
        self,
        expected_flow: Dict[str, Any],
        actual_flow: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Compare expected vs actual execution flows"""
        differences = []
        
        # Check for missing steps
        for step_id in expected_flow:
            if step_id not in actual_flow:
                differences.append({
                    'step_id': step_id,
                    'message': f"Expected step '{step_id}' was not executed",
                    'expected': 'executed',
                    'actual': 'not_executed'
                })
        
        # Check for unexpected steps
        for step_id in actual_flow:
            if step_id not in expected_flow:
                differences.append({
                    'step_id': step_id,
                    'message': f"Unexpected step '{step_id}' was executed",
                    'expected': 'not_executed',
                    'actual': 'executed'
                })
        
        return differences

# Global reconciliation engine instance
reconciliation_engine = DSLReconciliationEngine()

def get_reconciliation_engine() -> DSLReconciliationEngine:
    """Get the global reconciliation engine instance"""
    return reconciliation_engine
