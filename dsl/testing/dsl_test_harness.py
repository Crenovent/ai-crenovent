# DSL Test Harness and Validation Tools
# Tasks 6.4-T17, T22, T24: DSL test harness, linting tool, SLA metrics

import asyncio
import json
import yaml
import uuid
import time
from datetime import datetime, timezone
from typing import Dict, List, Any, Optional, Union, Tuple
from dataclasses import dataclass, asdict
from enum import Enum
import logging
import re

logger = logging.getLogger(__name__)

class DSLValidationSeverity(Enum):
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"

class DSLTestResult(Enum):
    PASSED = "passed"
    FAILED = "failed"
    SKIPPED = "skipped"
    ERROR = "error"

@dataclass
class DSLValidationIssue:
    """Individual DSL validation issue"""
    severity: DSLValidationSeverity
    message: str
    line_number: Optional[int]
    column_number: Optional[int]
    rule_id: str
    suggestion: Optional[str] = None
    context: Optional[str] = None

@dataclass
class DSLTestCase:
    """Individual DSL test case"""
    test_id: str
    test_name: str
    description: str
    dsl_content: str
    expected_result: DSLTestResult
    expected_issues: List[DSLValidationIssue]
    test_data: Dict[str, Any]
    timeout_seconds: int = 30

@dataclass
class DSLTestSuiteResult:
    """Result of DSL test suite execution"""
    suite_name: str
    total_tests: int
    passed_tests: int
    failed_tests: int
    skipped_tests: int
    error_tests: int
    execution_time_ms: int
    test_results: List['DSLTestCaseResult']
    sla_metrics: Dict[str, Any]
    executed_at: str

@dataclass
class DSLTestCaseResult:
    """Result of individual DSL test case"""
    test_id: str
    test_name: str
    result: DSLTestResult
    execution_time_ms: int
    validation_issues: List[DSLValidationIssue]
    error_message: Optional[str] = None
    actual_output: Optional[Dict[str, Any]] = None

class DSLLinter:
    """
    DSL Linting Tool for authoring validation
    Task 6.4-T22: Build linting tool for DSL authoring
    """
    
    def __init__(self):
        self.linting_rules = self._load_linting_rules()
        self.industry_rules = self._load_industry_specific_rules()
        
    def lint_dsl(
        self,
        dsl_content: str,
        industry_code: str = "SaaS",
        compliance_frameworks: List[str] = None
    ) -> List[DSLValidationIssue]:
        """Lint DSL content and return validation issues"""
        
        issues = []
        
        try:
            # Parse DSL content
            if isinstance(dsl_content, str):
                dsl_data = yaml.safe_load(dsl_content)
            else:
                dsl_data = dsl_content
            
            # Apply core linting rules
            issues.extend(self._apply_core_rules(dsl_data, dsl_content))
            
            # Apply industry-specific rules
            industry_rules = self.industry_rules.get(industry_code, [])
            issues.extend(self._apply_industry_rules(dsl_data, industry_rules))
            
            # Apply compliance framework rules
            if compliance_frameworks:
                issues.extend(self._apply_compliance_rules(dsl_data, compliance_frameworks))
            
            return sorted(issues, key=lambda x: (x.severity.value, x.line_number or 0))
            
        except yaml.YAMLError as e:
            return [DSLValidationIssue(
                severity=DSLValidationSeverity.CRITICAL,
                message=f"YAML parsing error: {str(e)}",
                line_number=getattr(e, 'problem_mark', {}).get('line'),
                column_number=getattr(e, 'problem_mark', {}).get('column'),
                rule_id="YAML_PARSE_ERROR",
                suggestion="Check YAML syntax and indentation"
            )]
        except Exception as e:
            return [DSLValidationIssue(
                severity=DSLValidationSeverity.CRITICAL,
                message=f"Linting error: {str(e)}",
                line_number=None,
                column_number=None,
                rule_id="LINTING_ERROR"
            )]
    
    def _apply_core_rules(
        self, 
        dsl_data: Dict[str, Any], 
        dsl_content: str
    ) -> List[DSLValidationIssue]:
        """Apply core DSL linting rules"""
        issues = []
        
        # Rule: Required fields
        required_fields = ['workflow_id', 'version', 'steps', 'governance']
        for field in required_fields:
            if field not in dsl_data:
                issues.append(DSLValidationIssue(
                    severity=DSLValidationSeverity.CRITICAL,
                    message=f"Missing required field: {field}",
                    line_number=1,
                    column_number=1,
                    rule_id="MISSING_REQUIRED_FIELD",
                    suggestion=f"Add '{field}:' to your workflow definition"
                ))
        
        # Rule: Workflow ID format
        workflow_id = dsl_data.get('workflow_id', '')
        if workflow_id and not re.match(r'^[a-z0-9_]+$', workflow_id):
            issues.append(DSLValidationIssue(
                severity=DSLValidationSeverity.ERROR,
                message="Workflow ID should contain only lowercase letters, numbers, and underscores",
                line_number=self._find_line_number(dsl_content, 'workflow_id'),
                column_number=None,
                rule_id="INVALID_WORKFLOW_ID_FORMAT",
                suggestion="Use format: 'my_workflow_name' or 'workflow_v1_0'"
            ))
        
        # Rule: Steps validation
        steps = dsl_data.get('steps', [])
        if not steps:
            issues.append(DSLValidationIssue(
                severity=DSLValidationSeverity.CRITICAL,
                message="Workflow must contain at least one step",
                line_number=self._find_line_number(dsl_content, 'steps'),
                column_number=None,
                rule_id="NO_STEPS_DEFINED",
                suggestion="Add at least one step to your workflow"
            ))
        
        # Rule: Step validation
        step_ids = set()
        for i, step in enumerate(steps):
            step_line = self._find_line_number(dsl_content, f"- id: {step.get('id', '')}")
            
            # Check required step fields
            if 'id' not in step:
                issues.append(DSLValidationIssue(
                    severity=DSLValidationSeverity.CRITICAL,
                    message=f"Step {i+1} missing required 'id' field",
                    line_number=step_line,
                    column_number=None,
                    rule_id="MISSING_STEP_ID",
                    suggestion="Add 'id: step_name' to your step definition"
                ))
            
            if 'type' not in step:
                issues.append(DSLValidationIssue(
                    severity=DSLValidationSeverity.CRITICAL,
                    message=f"Step {step.get('id', i+1)} missing required 'type' field",
                    line_number=step_line,
                    column_number=None,
                    rule_id="MISSING_STEP_TYPE",
                    suggestion="Add 'type: query|decision|notify|...' to your step"
                ))
            
            # Check for duplicate step IDs
            step_id = step.get('id')
            if step_id:
                if step_id in step_ids:
                    issues.append(DSLValidationIssue(
                        severity=DSLValidationSeverity.ERROR,
                        message=f"Duplicate step ID: {step_id}",
                        line_number=step_line,
                        column_number=None,
                        rule_id="DUPLICATE_STEP_ID",
                        suggestion=f"Use a unique ID for step {step_id}"
                    ))
                step_ids.add(step_id)
            
            # Check governance block
            if 'governance' not in step:
                issues.append(DSLValidationIssue(
                    severity=DSLValidationSeverity.ERROR,
                    message=f"Step {step.get('id', i+1)} missing governance block",
                    line_number=step_line,
                    column_number=None,
                    rule_id="MISSING_STEP_GOVERNANCE",
                    suggestion="Add governance block with policy_id and evidence_capture"
                ))
        
        # Rule: Governance validation
        governance = dsl_data.get('governance', {})
        if governance:
            required_governance_fields = ['tenant_id', 'industry_code', 'compliance_frameworks']
            for field in required_governance_fields:
                if field not in governance:
                    issues.append(DSLValidationIssue(
                        severity=DSLValidationSeverity.WARNING,
                        message=f"Governance missing recommended field: {field}",
                        line_number=self._find_line_number(dsl_content, 'governance'),
                        column_number=None,
                        rule_id="MISSING_GOVERNANCE_FIELD",
                        suggestion=f"Add '{field}:' to governance section"
                    ))
        
        return issues
    
    def _apply_industry_rules(
        self,
        dsl_data: Dict[str, Any],
        industry_rules: List[Dict[str, Any]]
    ) -> List[DSLValidationIssue]:
        """Apply industry-specific linting rules"""
        issues = []
        
        for rule in industry_rules:
            rule_result = self._evaluate_rule(dsl_data, rule)
            if rule_result:
                issues.append(rule_result)
        
        return issues
    
    def _apply_compliance_rules(
        self,
        dsl_data: Dict[str, Any],
        compliance_frameworks: List[str]
    ) -> List[DSLValidationIssue]:
        """Apply compliance framework-specific rules"""
        issues = []
        
        # SOX compliance rules
        if 'SOX_SAAS' in compliance_frameworks:
            if not dsl_data.get('governance', {}).get('evidence_retention_days'):
                issues.append(DSLValidationIssue(
                    severity=DSLValidationSeverity.ERROR,
                    message="SOX compliance requires evidence_retention_days to be specified",
                    line_number=None,
                    column_number=None,
                    rule_id="SOX_EVIDENCE_RETENTION_REQUIRED",
                    suggestion="Add 'evidence_retention_days: 2555' for 7-year SOX retention"
                ))
        
        # GDPR compliance rules
        if 'GDPR_SAAS' in compliance_frameworks:
            steps = dsl_data.get('steps', [])
            for step in steps:
                if step.get('type') == 'query' and not step.get('evidence', {}).get('pii_redaction'):
                    issues.append(DSLValidationIssue(
                        severity=DSLValidationSeverity.WARNING,
                        message=f"GDPR compliance recommends PII redaction for query step: {step.get('id')}",
                        line_number=None,
                        column_number=None,
                        rule_id="GDPR_PII_REDACTION_RECOMMENDED",
                        suggestion="Add 'pii_redaction: true' to evidence block"
                    ))
        
        return issues
    
    def _find_line_number(self, content: str, search_text: str) -> Optional[int]:
        """Find line number of text in content"""
        lines = content.split('\n')
        for i, line in enumerate(lines):
            if search_text in line:
                return i + 1
        return None
    
    def _load_linting_rules(self) -> Dict[str, Any]:
        """Load core linting rules"""
        return {
            'required_fields': ['workflow_id', 'version', 'steps', 'governance'],
            'step_types': ['query', 'decision', 'ml_decision', 'agent_call', 'notify', 'governance', 'assert'],
            'governance_fields': ['policy_id', 'evidence_capture'],
            'naming_patterns': {
                'workflow_id': r'^[a-z0-9_]+$',
                'step_id': r'^[a-z0-9_]+$'
            }
        }
    
    def _load_industry_specific_rules(self) -> Dict[str, List[Dict[str, Any]]]:
        """Load industry-specific linting rules"""
        return {
            'Banking': [
                {
                    'rule_id': 'BANKING_KYC_REQUIRED',
                    'condition': 'workflow contains loan or credit',
                    'requirement': 'KYC validation step required',
                    'severity': 'ERROR'
                }
            ],
            'Insurance': [
                {
                    'rule_id': 'INSURANCE_SOLVENCY_CHECK',
                    'condition': 'workflow contains claims',
                    'requirement': 'Solvency impact calculation required',
                    'severity': 'WARNING'
                }
            ]
        }
    
    def _evaluate_rule(self, dsl_data: Dict[str, Any], rule: Dict[str, Any]) -> Optional[DSLValidationIssue]:
        """Evaluate individual industry rule"""
        # Simplified rule evaluation - in practice would be more sophisticated
        return None

class DSLTestHarness:
    """
    Comprehensive DSL Test Harness
    Task 6.4-T17: Implement DSL test harness for developer validation
    """
    
    def __init__(self, dsl_parser=None, dsl_linter=None):
        self.dsl_parser = dsl_parser
        self.dsl_linter = dsl_linter or DSLLinter()
        self.test_suites = {}
        
    async def run_test_suite(
        self,
        suite_name: str,
        test_cases: List[DSLTestCase]
    ) -> DSLTestSuiteResult:
        """Run complete DSL test suite"""
        
        start_time = time.time()
        test_results = []
        
        logger.info(f"Running DSL test suite: {suite_name} ({len(test_cases)} tests)")
        
        for test_case in test_cases:
            result = await self._run_single_test(test_case)
            test_results.append(result)
        
        execution_time_ms = int((time.time() - start_time) * 1000)
        
        # Calculate results
        passed_tests = sum(1 for r in test_results if r.result == DSLTestResult.PASSED)
        failed_tests = sum(1 for r in test_results if r.result == DSLTestResult.FAILED)
        skipped_tests = sum(1 for r in test_results if r.result == DSLTestResult.SKIPPED)
        error_tests = sum(1 for r in test_results if r.result == DSLTestResult.ERROR)
        
        # Calculate SLA metrics (Task 6.4-T24)
        sla_metrics = self._calculate_sla_metrics(test_results, execution_time_ms)
        
        suite_result = DSLTestSuiteResult(
            suite_name=suite_name,
            total_tests=len(test_cases),
            passed_tests=passed_tests,
            failed_tests=failed_tests,
            skipped_tests=skipped_tests,
            error_tests=error_tests,
            execution_time_ms=execution_time_ms,
            test_results=test_results,
            sla_metrics=sla_metrics,
            executed_at=datetime.now(timezone.utc).isoformat()
        )
        
        logger.info(f"DSL test suite completed: {passed_tests}/{len(test_cases)} tests passed")
        return suite_result
    
    async def _run_single_test(self, test_case: DSLTestCase) -> DSLTestCaseResult:
        """Run individual DSL test case"""
        
        start_time = time.time()
        
        try:
            # Lint DSL content
            validation_issues = self.dsl_linter.lint_dsl(test_case.dsl_content)
            
            # Determine test result based on validation issues
            critical_issues = [i for i in validation_issues if i.severity == DSLValidationSeverity.CRITICAL]
            error_issues = [i for i in validation_issues if i.severity == DSLValidationSeverity.ERROR]
            
            if critical_issues and test_case.expected_result == DSLTestResult.PASSED:
                result = DSLTestResult.FAILED
            elif not critical_issues and not error_issues and test_case.expected_result == DSLTestResult.PASSED:
                result = DSLTestResult.PASSED
            elif test_case.expected_result == DSLTestResult.FAILED and (critical_issues or error_issues):
                result = DSLTestResult.PASSED  # Expected to fail and did fail
            else:
                result = DSLTestResult.FAILED
            
            execution_time_ms = int((time.time() - start_time) * 1000)
            
            return DSLTestCaseResult(
                test_id=test_case.test_id,
                test_name=test_case.test_name,
                result=result,
                execution_time_ms=execution_time_ms,
                validation_issues=validation_issues,
                actual_output={'validation_issues': len(validation_issues)}
            )
            
        except Exception as e:
            execution_time_ms = int((time.time() - start_time) * 1000)
            
            return DSLTestCaseResult(
                test_id=test_case.test_id,
                test_name=test_case.test_name,
                result=DSLTestResult.ERROR,
                execution_time_ms=execution_time_ms,
                validation_issues=[],
                error_message=str(e)
            )
    
    def _calculate_sla_metrics(
        self,
        test_results: List[DSLTestCaseResult],
        total_execution_time_ms: int
    ) -> Dict[str, Any]:
        """Calculate SLA metrics for DSL performance"""
        
        if not test_results:
            return {}
        
        execution_times = [r.execution_time_ms for r in test_results]
        
        return {
            'average_parse_time_ms': sum(execution_times) / len(execution_times),
            'max_parse_time_ms': max(execution_times),
            'min_parse_time_ms': min(execution_times),
            'total_execution_time_ms': total_execution_time_ms,
            'validation_success_rate': len([r for r in test_results if r.result == DSLTestResult.PASSED]) / len(test_results) * 100,
            'tests_per_second': len(test_results) / (total_execution_time_ms / 1000) if total_execution_time_ms > 0 else 0,
            'sla_compliance': {
                'parse_time_under_100ms': len([t for t in execution_times if t < 100]) / len(execution_times) * 100,
                'parse_time_under_500ms': len([t for t in execution_times if t < 500]) / len(execution_times) * 100,
                'validation_rate_above_95pct': len([r for r in test_results if r.result == DSLTestResult.PASSED]) / len(test_results) >= 0.95
            }
        }
    
    def create_sample_test_cases(self) -> List[DSLTestCase]:
        """Create sample test cases for DSL validation"""
        
        return [
            DSLTestCase(
                test_id="valid_saas_workflow",
                test_name="Valid SaaS Pipeline Hygiene Workflow",
                description="Test valid SaaS workflow with all required fields",
                dsl_content="""
workflow_id: "test_saas_pipeline"
version: "1.0"
industry_overlay: "SaaS"
compliance_frameworks: ["SOX_SAAS"]
steps:
  - id: "test_step"
    type: "query"
    params:
      data_source: "test"
    governance:
      policy_id: "test_policy"
      evidence_capture: true
governance:
  tenant_id: 1300
  industry_code: "SaaS"
  compliance_frameworks: ["SOX_SAAS"]
                """,
                expected_result=DSLTestResult.PASSED,
                expected_issues=[],
                test_data={}
            ),
            DSLTestCase(
                test_id="missing_workflow_id",
                test_name="Missing Workflow ID",
                description="Test workflow missing required workflow_id field",
                dsl_content="""
version: "1.0"
steps:
  - id: "test_step"
    type: "query"
governance:
  tenant_id: 1300
                """,
                expected_result=DSLTestResult.FAILED,
                expected_issues=[
                    DSLValidationIssue(
                        severity=DSLValidationSeverity.CRITICAL,
                        message="Missing required field: workflow_id",
                        line_number=1,
                        column_number=1,
                        rule_id="MISSING_REQUIRED_FIELD"
                    )
                ],
                test_data={}
            ),
            DSLTestCase(
                test_id="invalid_step_type",
                test_name="Invalid Step Type",
                description="Test workflow with invalid step type",
                dsl_content="""
workflow_id: "test_invalid_step"
version: "1.0"
steps:
  - id: "test_step"
    type: "invalid_type"
    governance:
      policy_id: "test_policy"
      evidence_capture: true
governance:
  tenant_id: 1300
                """,
                expected_result=DSLTestResult.FAILED,
                expected_issues=[],
                test_data={}
            )
        ]

# Global instances
dsl_linter = DSLLinter()
dsl_test_harness = DSLTestHarness(dsl_linter=dsl_linter)
