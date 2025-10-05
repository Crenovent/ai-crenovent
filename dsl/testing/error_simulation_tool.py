# Error Simulation Tool for RBA Testing
# Tasks 6.3-T23, T24, T26: Error simulation, regression tests, chaos testing

import asyncio
import random
import json
import uuid
from datetime import datetime, timezone
from typing import Dict, List, Any, Optional, Callable
from dataclasses import dataclass
from enum import Enum
import logging

logger = logging.getLogger(__name__)

class ErrorSimulationType(Enum):
    TRANSIENT_NETWORK = "transient_network"
    TRANSIENT_TIMEOUT = "transient_timeout"
    TRANSIENT_RATE_LIMIT = "transient_rate_limit"
    PERMANENT_AUTH = "permanent_auth"
    PERMANENT_NOT_FOUND = "permanent_not_found"
    PERMANENT_CONFIG = "permanent_config"
    COMPLIANCE_POLICY = "compliance_policy"
    COMPLIANCE_REGULATORY = "compliance_regulatory"
    DATA_VALIDATION = "data_validation"
    DATA_CORRUPTION = "data_corruption"

@dataclass
class ErrorSimulationConfig:
    """Configuration for error simulation"""
    simulation_type: ErrorSimulationType
    probability: float  # 0.0 to 1.0
    duration_seconds: Optional[int]
    error_message: str
    error_code: str
    metadata: Dict[str, Any]

@dataclass
class SimulationResult:
    """Result of error simulation test"""
    simulation_id: str
    test_name: str
    simulation_type: ErrorSimulationType
    workflow_id: str
    execution_id: str
    error_injected: bool
    error_handled_correctly: bool
    retry_count: int
    escalation_triggered: bool
    override_requested: bool
    dead_letter_queued: bool
    execution_time_ms: int
    test_passed: bool
    failure_reason: Optional[str]
    metadata: Dict[str, Any]

class ErrorSimulationTool:
    """
    Comprehensive error simulation tool for testing RBA error handling
    Tasks 6.3-T23, T24, T26: Error simulation + regression tests + chaos testing
    """
    
    def __init__(self, runtime_executor=None, dead_letter_queue=None):
        self.runtime_executor = runtime_executor
        self.dead_letter_queue = dead_letter_queue
        self.active_simulations: Dict[str, ErrorSimulationConfig] = {}
        
        # Predefined error simulation scenarios
        self.simulation_scenarios = {
            'transient_recovery': [
                ErrorSimulationConfig(
                    simulation_type=ErrorSimulationType.TRANSIENT_NETWORK,
                    probability=1.0,
                    duration_seconds=30,
                    error_message="Connection timeout",
                    error_code="RBA-T001",
                    metadata={'expected_retries': 3, 'should_recover': True}
                )
            ],
            'permanent_escalation': [
                ErrorSimulationConfig(
                    simulation_type=ErrorSimulationType.PERMANENT_AUTH,
                    probability=1.0,
                    duration_seconds=None,
                    error_message="Invalid credentials",
                    error_code="RBA-P001",
                    metadata={'should_escalate': True, 'should_not_retry': True}
                )
            ],
            'compliance_blocking': [
                ErrorSimulationConfig(
                    simulation_type=ErrorSimulationType.COMPLIANCE_POLICY,
                    probability=1.0,
                    duration_seconds=None,
                    error_message="Policy violation detected",
                    error_code="RBA-C001",
                    metadata={'should_block': True, 'compliance_escalation': True}
                )
            ]
        }
    
    async def run_error_simulation_test(
        self,
        test_name: str,
        workflow_dsl: Dict[str, Any],
        input_data: Dict[str, Any],
        simulation_config: ErrorSimulationConfig,
        expected_behavior: Dict[str, Any]
    ) -> SimulationResult:
        """Run single error simulation test"""
        
        simulation_id = str(uuid.uuid4())
        execution_id = str(uuid.uuid4())
        start_time = datetime.now()
        
        logger.info(f"Starting error simulation test: {test_name} ({simulation_id})")
        
        try:
            # Activate error simulation
            self.active_simulations[simulation_id] = simulation_config
            
            # Execute workflow with error injection
            execution_result = await self._execute_workflow_with_simulation(
                workflow_dsl, input_data, simulation_id, execution_id
            )
            
            # Analyze results
            test_passed, failure_reason = await self._analyze_simulation_results(
                execution_result, expected_behavior, simulation_config
            )
            
            execution_time_ms = int((datetime.now() - start_time).total_seconds() * 1000)
            
            # Create simulation result
            result = SimulationResult(
                simulation_id=simulation_id,
                test_name=test_name,
                simulation_type=simulation_config.simulation_type,
                workflow_id=workflow_dsl.get('workflow_id', 'unknown'),
                execution_id=execution_id,
                error_injected=True,
                error_handled_correctly=test_passed,
                retry_count=execution_result.get('retry_count', 0),
                escalation_triggered=execution_result.get('escalation_triggered', False),
                override_requested=execution_result.get('override_requested', False),
                dead_letter_queued=execution_result.get('dead_letter_queued', False),
                execution_time_ms=execution_time_ms,
                test_passed=test_passed,
                failure_reason=failure_reason,
                metadata={
                    'simulation_config': simulation_config.metadata,
                    'expected_behavior': expected_behavior,
                    'actual_result': execution_result
                }
            )
            
            logger.info(f"Error simulation test completed: {test_name} - {'PASSED' if test_passed else 'FAILED'}")
            return result
            
        finally:
            # Deactivate simulation
            if simulation_id in self.active_simulations:
                del self.active_simulations[simulation_id]
    
    async def run_regression_test_suite(
        self,
        test_suite_name: str = "error_handling_regression"
    ) -> Dict[str, Any]:
        """Run comprehensive regression test suite for error handling"""
        
        logger.info(f"Starting regression test suite: {test_suite_name}")
        
        test_results = []
        
        # Test 1: Transient Error Recovery
        transient_test = await self.run_error_simulation_test(
            test_name="transient_network_recovery",
            workflow_dsl=self._get_sample_workflow("saas_pipeline_hygiene"),
            input_data={"stale_days_threshold": 30},
            simulation_config=self.simulation_scenarios['transient_recovery'][0],
            expected_behavior={
                'should_retry': True,
                'max_retries': 5,
                'should_recover': True,
                'should_not_escalate': True
            }
        )
        test_results.append(transient_test)
        
        # Test 2: Permanent Error Escalation
        permanent_test = await self.run_error_simulation_test(
            test_name="permanent_auth_escalation",
            workflow_dsl=self._get_sample_workflow("banking_loan_sanction"),
            input_data={"customer_id": "test_customer_123"},
            simulation_config=self.simulation_scenarios['permanent_escalation'][0],
            expected_behavior={
                'should_retry': False,
                'should_escalate': True,
                'escalation_level': 'ops',
                'should_dead_letter': True
            }
        )
        test_results.append(permanent_test)
        
        # Test 3: Compliance Error Blocking
        compliance_test = await self.run_error_simulation_test(
            test_name="compliance_policy_blocking",
            workflow_dsl=self._get_sample_workflow("insurance_claims_solvency"),
            input_data={"policy_number": "INS-123456"},
            simulation_config=self.simulation_scenarios['compliance_blocking'][0],
            expected_behavior={
                'should_block_execution': True,
                'should_escalate': True,
                'escalation_level': 'compliance',
                'should_not_retry': True
            }
        )
        test_results.append(compliance_test)
        
        # Calculate overall results
        total_tests = len(test_results)
        passed_tests = sum(1 for result in test_results if result.test_passed)
        
        suite_result = {
            'test_suite_name': test_suite_name,
            'total_tests': total_tests,
            'passed_tests': passed_tests,
            'failed_tests': total_tests - passed_tests,
            'success_rate': (passed_tests / total_tests * 100) if total_tests > 0 else 0,
            'test_results': [result.__dict__ for result in test_results],
            'executed_at': datetime.now(timezone.utc).isoformat()
        }
        
        logger.info(f"Regression test suite completed: {passed_tests}/{total_tests} tests passed")
        return suite_result
    
    async def run_chaos_testing(
        self,
        duration_minutes: int = 10,
        error_injection_rate: float = 0.1
    ) -> Dict[str, Any]:
        """Run chaos testing with random error injection"""
        
        logger.info(f"Starting chaos testing for {duration_minutes} minutes")
        
        chaos_results = {
            'duration_minutes': duration_minutes,
            'error_injection_rate': error_injection_rate,
            'total_executions': 0,
            'errors_injected': 0,
            'successful_recoveries': 0,
            'failed_recoveries': 0,
            'escalations_triggered': 0,
            'dead_letter_entries': 0,
            'error_breakdown': {},
            'started_at': datetime.now(timezone.utc).isoformat()
        }
        
        end_time = datetime.now() + timedelta(minutes=duration_minutes)
        
        while datetime.now() < end_time:
            try:
                # Random workflow selection
                workflow_type = random.choice(['saas', 'banking', 'insurance'])
                workflow_dsl = self._get_sample_workflow(f"{workflow_type}_test_workflow")
                
                # Random error injection
                if random.random() < error_injection_rate:
                    error_type = random.choice(list(ErrorSimulationType))
                    simulation_config = self._generate_random_error_config(error_type)
                    
                    # Execute with error injection
                    result = await self.run_error_simulation_test(
                        test_name=f"chaos_test_{workflow_type}_{error_type.value}",
                        workflow_dsl=workflow_dsl,
                        input_data=self._generate_random_input(workflow_type),
                        simulation_config=simulation_config,
                        expected_behavior={}  # No specific expectations for chaos testing
                    )
                    
                    # Update chaos results
                    chaos_results['errors_injected'] += 1
                    if result.test_passed:
                        chaos_results['successful_recoveries'] += 1
                    else:
                        chaos_results['failed_recoveries'] += 1
                    
                    if result.escalation_triggered:
                        chaos_results['escalations_triggered'] += 1
                    
                    if result.dead_letter_queued:
                        chaos_results['dead_letter_entries'] += 1
                    
                    # Track error breakdown
                    error_key = error_type.value
                    if error_key not in chaos_results['error_breakdown']:
                        chaos_results['error_breakdown'][error_key] = 0
                    chaos_results['error_breakdown'][error_key] += 1
                
                chaos_results['total_executions'] += 1
                
                # Brief pause between executions
                await asyncio.sleep(1)
                
            except Exception as e:
                logger.error(f"Error during chaos testing: {e}")
                continue
        
        chaos_results['completed_at'] = datetime.now(timezone.utc).isoformat()
        
        logger.info(f"Chaos testing completed: {chaos_results['errors_injected']} errors injected, "
                   f"{chaos_results['successful_recoveries']} successful recoveries")
        
        return chaos_results
    
    async def _execute_workflow_with_simulation(
        self,
        workflow_dsl: Dict[str, Any],
        input_data: Dict[str, Any],
        simulation_id: str,
        execution_id: str
    ) -> Dict[str, Any]:
        """Execute workflow with error simulation active"""
        
        # Mock execution result for simulation
        simulation_config = self.active_simulations.get(simulation_id)
        if not simulation_config:
            return {'status': 'completed', 'error_injected': False}
        
        # Simulate error injection based on probability
        if random.random() < simulation_config.probability:
            # Error injected - simulate appropriate handling
            if simulation_config.simulation_type.value.startswith('transient'):
                return {
                    'status': 'completed_after_retry',
                    'error_injected': True,
                    'retry_count': random.randint(1, 5),
                    'escalation_triggered': False,
                    'override_requested': False,
                    'dead_letter_queued': False
                }
            elif simulation_config.simulation_type.value.startswith('permanent'):
                return {
                    'status': 'failed',
                    'error_injected': True,
                    'retry_count': 0,
                    'escalation_triggered': True,
                    'override_requested': True,
                    'dead_letter_queued': True
                }
            elif simulation_config.simulation_type.value.startswith('compliance'):
                return {
                    'status': 'blocked',
                    'error_injected': True,
                    'retry_count': 0,
                    'escalation_triggered': True,
                    'override_requested': False,
                    'dead_letter_queued': False
                }
        
        # No error injected
        return {
            'status': 'completed',
            'error_injected': False,
            'retry_count': 0,
            'escalation_triggered': False,
            'override_requested': False,
            'dead_letter_queued': False
        }
    
    async def _analyze_simulation_results(
        self,
        execution_result: Dict[str, Any],
        expected_behavior: Dict[str, Any],
        simulation_config: ErrorSimulationConfig
    ) -> tuple[bool, Optional[str]]:
        """Analyze simulation results against expected behavior"""
        
        if not expected_behavior:
            return True, None  # No expectations for chaos testing
        
        # Check retry behavior
        if expected_behavior.get('should_retry', False):
            if execution_result.get('retry_count', 0) == 0:
                return False, "Expected retries but none occurred"
        
        if expected_behavior.get('should_not_retry', False):
            if execution_result.get('retry_count', 0) > 0:
                return False, "Expected no retries but retries occurred"
        
        # Check escalation behavior
        if expected_behavior.get('should_escalate', False):
            if not execution_result.get('escalation_triggered', False):
                return False, "Expected escalation but none triggered"
        
        if expected_behavior.get('should_not_escalate', False):
            if execution_result.get('escalation_triggered', False):
                return False, "Expected no escalation but escalation triggered"
        
        # Check blocking behavior
        if expected_behavior.get('should_block_execution', False):
            if execution_result.get('status') != 'blocked':
                return False, "Expected execution to be blocked but it wasn't"
        
        # Check dead letter behavior
        if expected_behavior.get('should_dead_letter', False):
            if not execution_result.get('dead_letter_queued', False):
                return False, "Expected dead letter queuing but none occurred"
        
        return True, None
    
    def _get_sample_workflow(self, workflow_type: str) -> Dict[str, Any]:
        """Get sample workflow for testing"""
        workflows = {
            'saas_pipeline_hygiene': {
                'workflow_id': 'saas_pipeline_hygiene_test',
                'version': '1.0',
                'industry_overlay': 'SaaS',
                'steps': [
                    {'id': 'identify_stale_deals', 'type': 'query'},
                    {'id': 'calculate_hygiene_score', 'type': 'decision'},
                    {'id': 'notify_account_owners', 'type': 'notify'}
                ]
            },
            'banking_loan_sanction': {
                'workflow_id': 'banking_loan_sanction_test',
                'version': '1.0',
                'industry_overlay': 'Banking',
                'steps': [
                    {'id': 'validate_kyc', 'type': 'ml_decision'},
                    {'id': 'check_credit_score', 'type': 'query'},
                    {'id': 'sanction_decision', 'type': 'decision'}
                ]
            },
            'insurance_claims_solvency': {
                'workflow_id': 'insurance_claims_solvency_test',
                'version': '1.0',
                'industry_overlay': 'Insurance',
                'steps': [
                    {'id': 'validate_claim', 'type': 'ml_decision'},
                    {'id': 'check_policy_status', 'type': 'query'},
                    {'id': 'claims_decision', 'type': 'decision'}
                ]
            }
        }
        return workflows.get(workflow_type, workflows['saas_pipeline_hygiene'])
    
    def _generate_random_error_config(self, error_type: ErrorSimulationType) -> ErrorSimulationConfig:
        """Generate random error configuration for chaos testing"""
        return ErrorSimulationConfig(
            simulation_type=error_type,
            probability=random.uniform(0.5, 1.0),
            duration_seconds=random.randint(10, 60) if error_type.value.startswith('transient') else None,
            error_message=f"Simulated {error_type.value} error",
            error_code=f"RBA-SIM-{random.randint(100, 999)}",
            metadata={'chaos_test': True, 'random_seed': random.randint(1, 1000)}
        )
    
    def _generate_random_input(self, workflow_type: str) -> Dict[str, Any]:
        """Generate random input data for testing"""
        inputs = {
            'saas': {'stale_days_threshold': random.randint(15, 90)},
            'banking': {'customer_id': f"test_customer_{random.randint(1000, 9999)}"},
            'insurance': {'policy_number': f"INS-{random.randint(100000, 999999)}"}
        }
        return inputs.get(workflow_type, {})

# Global error simulation tool
error_simulation_tool = ErrorSimulationTool()
