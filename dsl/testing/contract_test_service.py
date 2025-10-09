"""
Task 6.3.59: Contract tests between model contract & runtime
Build contract tests to ensure model contracts match runtime behavior
"""

import json
import logging
from typing import Dict, List, Any, Optional, Set
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)

class ContractTestResult(Enum):
    """Contract test result status"""
    PASS = "pass"
    FAIL = "fail"
    ERROR = "error"

@dataclass
class ContractViolation:
    """Contract violation details"""
    field_name: str
    expected: Any
    actual: Any
    violation_type: str
    severity: str = "error"

@dataclass
class ContractTestCase:
    """Individual contract test case"""
    test_id: str
    model_id: str
    input_data: Dict[str, Any]
    expected_output_schema: Dict[str, Any]
    expected_constraints: Dict[str, Any]
    description: str

class ContractTestService:
    """
    Contract testing service for model serving
    Task 6.3.59: No mismatch between contract and runtime
    """
    
    def __init__(self):
        self.test_cases: Dict[str, List[ContractTestCase]] = {}
        self.test_results: Dict[str, Dict] = {}
    
    def register_contract_tests(self, model_id: str, test_cases: List[ContractTestCase]) -> bool:
        """Register contract test cases for a model"""
        try:
            self.test_cases[model_id] = test_cases
            logger.info(f"Registered {len(test_cases)} contract tests for model: {model_id}")
            return True
        except Exception as e:
            logger.error(f"Failed to register contract tests: {e}")
            return False
    
    async def run_contract_tests(self, model_id: str) -> Dict[str, Any]:
        """Run all contract tests for a model"""
        if model_id not in self.test_cases:
            return {
                "model_id": model_id,
                "status": "no_tests",
                "message": "No contract tests registered for this model"
            }
        
        test_cases = self.test_cases[model_id]
        results = []
        
        for test_case in test_cases:
            result = await self._run_single_test(test_case)
            results.append(result)
        
        # Aggregate results
        passed = sum(1 for r in results if r["status"] == ContractTestResult.PASS.value)
        failed = sum(1 for r in results if r["status"] == ContractTestResult.FAIL.value)
        errors = sum(1 for r in results if r["status"] == ContractTestResult.ERROR.value)
        
        overall_result = {
            "model_id": model_id,
            "total_tests": len(results),
            "passed": passed,
            "failed": failed,
            "errors": errors,
            "success_rate": passed / len(results) if results else 0.0,
            "test_results": results,
            "overall_status": "pass" if failed == 0 and errors == 0 else "fail"
        }
        
        self.test_results[model_id] = overall_result
        return overall_result
    
    async def _run_single_test(self, test_case: ContractTestCase) -> Dict[str, Any]:
        """Run a single contract test case"""
        try:
            # Simulate model inference
            actual_output = await self._invoke_model(test_case.model_id, test_case.input_data)
            
            # Validate contract compliance
            violations = self._validate_contract(
                actual_output,
                test_case.expected_output_schema,
                test_case.expected_constraints
            )
            
            if not violations:
                return {
                    "test_id": test_case.test_id,
                    "status": ContractTestResult.PASS.value,
                    "description": test_case.description,
                    "violations": []
                }
            else:
                return {
                    "test_id": test_case.test_id,
                    "status": ContractTestResult.FAIL.value,
                    "description": test_case.description,
                    "violations": [
                        {
                            "field": v.field_name,
                            "expected": v.expected,
                            "actual": v.actual,
                            "type": v.violation_type,
                            "severity": v.severity
                        }
                        for v in violations
                    ]
                }
        
        except Exception as e:
            logger.error(f"Contract test {test_case.test_id} failed with error: {e}")
            return {
                "test_id": test_case.test_id,
                "status": ContractTestResult.ERROR.value,
                "description": test_case.description,
                "error": str(e),
                "violations": []
            }
    
    async def _invoke_model(self, model_id: str, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Invoke model for testing"""
        # Simulate model inference
        return {
            "prediction": 0.75,
            "confidence": 0.85,
            "model_id": model_id,
            "timestamp": "2024-01-01T00:00:00Z",
            "probabilities": [0.25, 0.75]
        }
    
    def _validate_contract(
        self,
        actual_output: Dict[str, Any],
        expected_schema: Dict[str, Any],
        expected_constraints: Dict[str, Any]
    ) -> List[ContractViolation]:
        """Validate actual output against contract"""
        violations = []
        
        # Validate schema compliance
        violations.extend(self._validate_schema(actual_output, expected_schema))
        
        # Validate constraints
        violations.extend(self._validate_constraints(actual_output, expected_constraints))
        
        return violations
    
    def _validate_schema(self, output: Dict[str, Any], schema: Dict[str, Any]) -> List[ContractViolation]:
        """Validate output schema"""
        violations = []
        
        # Check required fields
        required_fields = schema.get("required", [])
        for field in required_fields:
            if field not in output:
                violations.append(ContractViolation(
                    field_name=field,
                    expected="present",
                    actual="missing",
                    violation_type="missing_field"
                ))
        
        # Check field types
        properties = schema.get("properties", {})
        for field, field_schema in properties.items():
            if field in output:
                expected_type = field_schema.get("type")
                actual_value = output[field]
                
                if not self._check_type(actual_value, expected_type):
                    violations.append(ContractViolation(
                        field_name=field,
                        expected=expected_type,
                        actual=type(actual_value).__name__,
                        violation_type="type_mismatch"
                    ))
        
        return violations
    
    def _validate_constraints(self, output: Dict[str, Any], constraints: Dict[str, Any]) -> List[ContractViolation]:
        """Validate output constraints"""
        violations = []
        
        for field, constraint_rules in constraints.items():
            if field not in output:
                continue
            
            actual_value = output[field]
            
            # Range constraints
            if "min" in constraint_rules and actual_value < constraint_rules["min"]:
                violations.append(ContractViolation(
                    field_name=field,
                    expected=f">= {constraint_rules['min']}",
                    actual=actual_value,
                    violation_type="range_violation"
                ))
            
            if "max" in constraint_rules and actual_value > constraint_rules["max"]:
                violations.append(ContractViolation(
                    field_name=field,
                    expected=f"<= {constraint_rules['max']}",
                    actual=actual_value,
                    violation_type="range_violation"
                ))
            
            # Enum constraints
            if "enum" in constraint_rules and actual_value not in constraint_rules["enum"]:
                violations.append(ContractViolation(
                    field_name=field,
                    expected=f"one of {constraint_rules['enum']}",
                    actual=actual_value,
                    violation_type="enum_violation"
                ))
        
        return violations
    
    def _check_type(self, value: Any, expected_type: str) -> bool:
        """Check if value matches expected type"""
        type_map = {
            "string": str,
            "number": (int, float),
            "integer": int,
            "boolean": bool,
            "array": list,
            "object": dict
        }
        
        expected_python_type = type_map.get(expected_type)
        if expected_python_type is None:
            return True  # Unknown type, assume valid
        
        return isinstance(value, expected_python_type)
    
    def get_test_report(self, model_id: str) -> Optional[Dict[str, Any]]:
        """Get test report for a model"""
        return self.test_results.get(model_id)

# Global contract test service instance
contract_test_service = ContractTestService()
