"""
Task 7.3-T26: Enforce Lint/Quality Gates at publish
Fail-closed validation with compiler + linters for baseline quality
"""

import asyncio
import json
import re
import uuid
from datetime import datetime, timezone
from enum import Enum
from typing import Dict, List, Optional, Any, Set
from dataclasses import dataclass, field

import yaml
from pathlib import Path


class QualityGateType(Enum):
    """Types of quality gates"""
    SYNTAX_VALIDATION = "syntax_validation"
    SCHEMA_COMPLIANCE = "schema_compliance"
    SECURITY_SCAN = "security_scan"
    PERFORMANCE_CHECK = "performance_check"
    DOCUMENTATION_CHECK = "documentation_check"
    NAMING_CONVENTION = "naming_convention"
    COMPLEXITY_ANALYSIS = "complexity_analysis"
    DEPENDENCY_CHECK = "dependency_check"


class QualitySeverity(Enum):
    """Quality issue severity levels"""
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    INFO = "info"


class QualityGateStatus(Enum):
    """Quality gate execution status"""
    PASSED = "passed"
    FAILED = "failed"
    WARNING = "warning"
    SKIPPED = "skipped"
    ERROR = "error"


@dataclass
class QualityIssue:
    """Individual quality issue"""
    issue_id: str
    gate_type: QualityGateType
    severity: QualitySeverity
    message: str
    file_path: Optional[str] = None
    line_number: Optional[int] = None
    column_number: Optional[int] = None
    rule_id: Optional[str] = None
    suggestion: Optional[str] = None
    auto_fixable: bool = False
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "issue_id": self.issue_id,
            "gate_type": self.gate_type.value,
            "severity": self.severity.value,
            "message": self.message,
            "file_path": self.file_path,
            "line_number": self.line_number,
            "column_number": self.column_number,
            "rule_id": self.rule_id,
            "suggestion": self.suggestion,
            "auto_fixable": self.auto_fixable
        }


@dataclass
class QualityGateResult:
    """Result of a quality gate execution"""
    gate_type: QualityGateType
    status: QualityGateStatus
    execution_time_ms: int
    issues: List[QualityIssue] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def critical_issues(self) -> List[QualityIssue]:
        return [issue for issue in self.issues if issue.severity == QualitySeverity.CRITICAL]
    
    @property
    def high_issues(self) -> List[QualityIssue]:
        return [issue for issue in self.issues if issue.severity == QualitySeverity.HIGH]
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "gate_type": self.gate_type.value,
            "status": self.status.value,
            "execution_time_ms": self.execution_time_ms,
            "issues": [issue.to_dict() for issue in self.issues],
            "metadata": self.metadata,
            "critical_count": len(self.critical_issues),
            "high_count": len(self.high_issues),
            "total_issues": len(self.issues)
        }


class QualityGateEngine:
    """Quality gate enforcement engine"""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or self._get_default_config()
        self.gates = self._initialize_gates()
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Get default quality gate configuration"""
        return {
            "fail_on_critical": True,
            "fail_on_high": True,
            "max_issues_per_gate": 100,
            "timeout_seconds": 300,
            "parallel_execution": True,
            "gates": {
                "syntax_validation": {"enabled": True, "timeout": 30},
                "schema_compliance": {"enabled": True, "timeout": 60},
                "security_scan": {"enabled": True, "timeout": 120},
                "performance_check": {"enabled": True, "timeout": 90},
                "documentation_check": {"enabled": True, "timeout": 30},
                "naming_convention": {"enabled": True, "timeout": 15},
                "complexity_analysis": {"enabled": True, "timeout": 45},
                "dependency_check": {"enabled": True, "timeout": 60}
            },
            "naming_rules": {
                "workflow_name_pattern": r"^[a-z][a-z0-9_]*[a-z0-9]$",
                "step_id_pattern": r"^[a-z][a-z0-9_]*$",
                "variable_pattern": r"^[a-z][a-z0-9_]*$",
                "max_workflow_name_length": 100,
                "max_step_count": 50,
                "max_nesting_depth": 5
            },
            "security_rules": {
                "no_hardcoded_secrets": True,
                "no_sql_injection_patterns": True,
                "no_unsafe_eval": True,
                "require_input_validation": True,
                "max_external_calls": 10
            }
        }
    
    def _initialize_gates(self) -> Dict[QualityGateType, Any]:
        """Initialize quality gate implementations"""
        return {
            QualityGateType.SYNTAX_VALIDATION: self._syntax_validator,
            QualityGateType.SCHEMA_COMPLIANCE: self._schema_validator,
            QualityGateType.SECURITY_SCAN: self._security_scanner,
            QualityGateType.PERFORMANCE_CHECK: self._performance_checker,
            QualityGateType.DOCUMENTATION_CHECK: self._documentation_checker,
            QualityGateType.NAMING_CONVENTION: self._naming_validator,
            QualityGateType.COMPLEXITY_ANALYSIS: self._complexity_analyzer,
            QualityGateType.DEPENDENCY_CHECK: self._dependency_checker
        }
    
    async def validate_workflow(
        self,
        workflow_content: str,
        workflow_metadata: Dict[str, Any],
        artifacts: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Validate workflow against all quality gates"""
        
        start_time = datetime.now(timezone.utc)
        results = {}
        overall_status = QualityGateStatus.PASSED
        
        # Parse workflow
        try:
            if workflow_content.strip().startswith('{'):
                workflow_data = json.loads(workflow_content)
            else:
                workflow_data = yaml.safe_load(workflow_content)
        except Exception as e:
            return {
                "overall_status": QualityGateStatus.FAILED.value,
                "error": f"Failed to parse workflow: {str(e)}",
                "execution_time_ms": 0,
                "results": {}
            }
        
        # Execute quality gates
        if self.config.get("parallel_execution", True):
            tasks = []
            for gate_type in QualityGateType:
                if self._is_gate_enabled(gate_type):
                    task = self._execute_gate(gate_type, workflow_data, workflow_metadata, artifacts)
                    tasks.append(task)
            
            gate_results = await asyncio.gather(*tasks, return_exceptions=True)
            
            for i, gate_type in enumerate(QualityGateType):
                if self._is_gate_enabled(gate_type):
                    result = gate_results[i] if i < len(gate_results) else None
                    if isinstance(result, Exception):
                        results[gate_type.value] = QualityGateResult(
                            gate_type=gate_type,
                            status=QualityGateStatus.ERROR,
                            execution_time_ms=0,
                            issues=[QualityIssue(
                                issue_id=str(uuid.uuid4()),
                                gate_type=gate_type,
                                severity=QualitySeverity.CRITICAL,
                                message=f"Gate execution failed: {str(result)}"
                            )]
                        ).to_dict()
                    else:
                        results[gate_type.value] = result.to_dict() if result else {}
        else:
            # Sequential execution
            for gate_type in QualityGateType:
                if self._is_gate_enabled(gate_type):
                    result = await self._execute_gate(gate_type, workflow_data, workflow_metadata, artifacts)
                    results[gate_type.value] = result.to_dict()
        
        # Determine overall status
        for result in results.values():
            if result.get("status") == QualityGateStatus.FAILED.value:
                overall_status = QualityGateStatus.FAILED
                break
            elif result.get("status") == QualityGateStatus.WARNING.value and overall_status == QualityGateStatus.PASSED:
                overall_status = QualityGateStatus.WARNING
        
        # Check fail conditions
        if self.config.get("fail_on_critical", True):
            for result in results.values():
                if result.get("critical_count", 0) > 0:
                    overall_status = QualityGateStatus.FAILED
                    break
        
        if self.config.get("fail_on_high", True) and overall_status != QualityGateStatus.FAILED:
            for result in results.values():
                if result.get("high_count", 0) > 0:
                    overall_status = QualityGateStatus.FAILED
                    break
        
        execution_time = (datetime.now(timezone.utc) - start_time).total_seconds() * 1000
        
        return {
            "overall_status": overall_status.value,
            "execution_time_ms": int(execution_time),
            "results": results,
            "summary": self._generate_summary(results),
            "recommendations": self._generate_recommendations(results)
        }
    
    def _is_gate_enabled(self, gate_type: QualityGateType) -> bool:
        """Check if a quality gate is enabled"""
        return self.config.get("gates", {}).get(gate_type.value, {}).get("enabled", True)
    
    async def _execute_gate(
        self,
        gate_type: QualityGateType,
        workflow_data: Dict[str, Any],
        metadata: Dict[str, Any],
        artifacts: Optional[Dict[str, Any]]
    ) -> QualityGateResult:
        """Execute a single quality gate"""
        
        start_time = datetime.now(timezone.utc)
        gate_func = self.gates.get(gate_type)
        
        if not gate_func:
            return QualityGateResult(
                gate_type=gate_type,
                status=QualityGateStatus.SKIPPED,
                execution_time_ms=0
            )
        
        try:
            issues = await gate_func(workflow_data, metadata, artifacts)
            status = QualityGateStatus.PASSED
            
            # Determine status based on issues
            if any(issue.severity == QualitySeverity.CRITICAL for issue in issues):
                status = QualityGateStatus.FAILED
            elif any(issue.severity == QualitySeverity.HIGH for issue in issues):
                status = QualityGateStatus.WARNING if not self.config.get("fail_on_high") else QualityGateStatus.FAILED
            
            execution_time = (datetime.now(timezone.utc) - start_time).total_seconds() * 1000
            
            return QualityGateResult(
                gate_type=gate_type,
                status=status,
                execution_time_ms=int(execution_time),
                issues=issues
            )
            
        except Exception as e:
            execution_time = (datetime.now(timezone.utc) - start_time).total_seconds() * 1000
            return QualityGateResult(
                gate_type=gate_type,
                status=QualityGateStatus.ERROR,
                execution_time_ms=int(execution_time),
                issues=[QualityIssue(
                    issue_id=str(uuid.uuid4()),
                    gate_type=gate_type,
                    severity=QualitySeverity.CRITICAL,
                    message=f"Gate execution error: {str(e)}"
                )]
            )
    
    async def _syntax_validator(self, workflow_data: Dict[str, Any], metadata: Dict[str, Any], artifacts: Optional[Dict[str, Any]]) -> List[QualityIssue]:
        """Validate DSL syntax"""
        issues = []
        
        # Check required fields
        required_fields = ["workflow_name", "steps"]
        for field in required_fields:
            if field not in workflow_data:
                issues.append(QualityIssue(
                    issue_id=str(uuid.uuid4()),
                    gate_type=QualityGateType.SYNTAX_VALIDATION,
                    severity=QualitySeverity.CRITICAL,
                    message=f"Missing required field: {field}",
                    rule_id="SYNTAX_001"
                ))
        
        # Validate steps structure
        steps = workflow_data.get("steps", [])
        if not isinstance(steps, list):
            issues.append(QualityIssue(
                issue_id=str(uuid.uuid4()),
                gate_type=QualityGateType.SYNTAX_VALIDATION,
                severity=QualitySeverity.CRITICAL,
                message="Steps must be a list",
                rule_id="SYNTAX_002"
            ))
        else:
            for i, step in enumerate(steps):
                if not isinstance(step, dict):
                    issues.append(QualityIssue(
                        issue_id=str(uuid.uuid4()),
                        gate_type=QualityGateType.SYNTAX_VALIDATION,
                        severity=QualitySeverity.CRITICAL,
                        message=f"Step {i} must be an object",
                        rule_id="SYNTAX_003"
                    ))
                    continue
                
                # Check required step fields
                step_required = ["id", "type"]
                for field in step_required:
                    if field not in step:
                        issues.append(QualityIssue(
                            issue_id=str(uuid.uuid4()),
                            gate_type=QualityGateType.SYNTAX_VALIDATION,
                            severity=QualitySeverity.HIGH,
                            message=f"Step {i} missing required field: {field}",
                            rule_id="SYNTAX_004"
                        ))
        
        return issues
    
    async def _schema_validator(self, workflow_data: Dict[str, Any], metadata: Dict[str, Any], artifacts: Optional[Dict[str, Any]]) -> List[QualityIssue]:
        """Validate schema compliance"""
        issues = []
        
        # Validate step types
        valid_step_types = {"query", "decision", "ml_decision", "agent_call", "notify", "governance"}
        steps = workflow_data.get("steps", [])
        
        for i, step in enumerate(steps):
            if isinstance(step, dict):
                step_type = step.get("type")
                if step_type and step_type not in valid_step_types:
                    issues.append(QualityIssue(
                        issue_id=str(uuid.uuid4()),
                        gate_type=QualityGateType.SCHEMA_COMPLIANCE,
                        severity=QualitySeverity.HIGH,
                        message=f"Step {i} has invalid type: {step_type}",
                        rule_id="SCHEMA_001",
                        suggestion=f"Use one of: {', '.join(valid_step_types)}"
                    ))
        
        return issues
    
    async def _security_scanner(self, workflow_data: Dict[str, Any], metadata: Dict[str, Any], artifacts: Optional[Dict[str, Any]]) -> List[QualityIssue]:
        """Scan for security issues"""
        issues = []
        
        # Convert workflow to string for pattern matching
        workflow_str = json.dumps(workflow_data, indent=2)
        
        # Check for hardcoded secrets
        secret_patterns = [
            (r'password\s*[:=]\s*["\'][^"\']{8,}["\']', "Potential hardcoded password"),
            (r'api[_-]?key\s*[:=]\s*["\'][^"\']{16,}["\']', "Potential hardcoded API key"),
            (r'secret\s*[:=]\s*["\'][^"\']{16,}["\']', "Potential hardcoded secret"),
            (r'token\s*[:=]\s*["\'][^"\']{20,}["\']', "Potential hardcoded token")
        ]
        
        for pattern, message in secret_patterns:
            matches = re.finditer(pattern, workflow_str, re.IGNORECASE)
            for match in matches:
                issues.append(QualityIssue(
                    issue_id=str(uuid.uuid4()),
                    gate_type=QualityGateType.SECURITY_SCAN,
                    severity=QualitySeverity.CRITICAL,
                    message=message,
                    rule_id="SEC_001",
                    suggestion="Use environment variables or secure parameter store"
                ))
        
        # Check for SQL injection patterns
        sql_patterns = [
            (r'SELECT\s+.*\s+FROM\s+.*\s+WHERE\s+.*\+', "Potential SQL injection via string concatenation"),
            (r'INSERT\s+INTO\s+.*\s+VALUES\s*\(.*\+', "Potential SQL injection in INSERT"),
            (r'UPDATE\s+.*\s+SET\s+.*\+', "Potential SQL injection in UPDATE")
        ]
        
        for pattern, message in sql_patterns:
            matches = re.finditer(pattern, workflow_str, re.IGNORECASE)
            for match in matches:
                issues.append(QualityIssue(
                    issue_id=str(uuid.uuid4()),
                    gate_type=QualityGateType.SECURITY_SCAN,
                    severity=QualitySeverity.HIGH,
                    message=message,
                    rule_id="SEC_002",
                    suggestion="Use parameterized queries"
                ))
        
        return issues
    
    async def _performance_checker(self, workflow_data: Dict[str, Any], metadata: Dict[str, Any], artifacts: Optional[Dict[str, Any]]) -> List[QualityIssue]:
        """Check performance characteristics"""
        issues = []
        
        steps = workflow_data.get("steps", [])
        
        # Check step count
        max_steps = self.config.get("naming_rules", {}).get("max_step_count", 50)
        if len(steps) > max_steps:
            issues.append(QualityIssue(
                issue_id=str(uuid.uuid4()),
                gate_type=QualityGateType.PERFORMANCE_CHECK,
                severity=QualitySeverity.MEDIUM,
                message=f"Workflow has {len(steps)} steps, exceeds recommended maximum of {max_steps}",
                rule_id="PERF_001",
                suggestion="Consider breaking into smaller workflows"
            ))
        
        # Check for potential infinite loops
        step_ids = set()
        for step in steps:
            if isinstance(step, dict):
                step_id = step.get("id")
                if step_id:
                    if step_id in step_ids:
                        issues.append(QualityIssue(
                            issue_id=str(uuid.uuid4()),
                            gate_type=QualityGateType.PERFORMANCE_CHECK,
                            severity=QualitySeverity.HIGH,
                            message=f"Duplicate step ID: {step_id}",
                            rule_id="PERF_002"
                        ))
                    step_ids.add(step_id)
        
        return issues
    
    async def _documentation_checker(self, workflow_data: Dict[str, Any], metadata: Dict[str, Any], artifacts: Optional[Dict[str, Any]]) -> List[QualityIssue]:
        """Check documentation completeness"""
        issues = []
        
        # Check workflow description
        if not workflow_data.get("description"):
            issues.append(QualityIssue(
                issue_id=str(uuid.uuid4()),
                gate_type=QualityGateType.DOCUMENTATION_CHECK,
                severity=QualitySeverity.MEDIUM,
                message="Workflow missing description",
                rule_id="DOC_001",
                suggestion="Add a clear description of workflow purpose"
            ))
        
        # Check step descriptions
        steps = workflow_data.get("steps", [])
        for i, step in enumerate(steps):
            if isinstance(step, dict) and not step.get("description"):
                issues.append(QualityIssue(
                    issue_id=str(uuid.uuid4()),
                    gate_type=QualityGateType.DOCUMENTATION_CHECK,
                    severity=QualitySeverity.LOW,
                    message=f"Step {i} ({step.get('id', 'unknown')}) missing description",
                    rule_id="DOC_002"
                ))
        
        return issues
    
    async def _naming_validator(self, workflow_data: Dict[str, Any], metadata: Dict[str, Any], artifacts: Optional[Dict[str, Any]]) -> List[QualityIssue]:
        """Validate naming conventions"""
        issues = []
        
        naming_rules = self.config.get("naming_rules", {})
        
        # Validate workflow name
        workflow_name = workflow_data.get("workflow_name", "")
        name_pattern = naming_rules.get("workflow_name_pattern", r"^[a-z][a-z0-9_]*[a-z0-9]$")
        max_length = naming_rules.get("max_workflow_name_length", 100)
        
        if not re.match(name_pattern, workflow_name):
            issues.append(QualityIssue(
                issue_id=str(uuid.uuid4()),
                gate_type=QualityGateType.NAMING_CONVENTION,
                severity=QualitySeverity.MEDIUM,
                message=f"Workflow name '{workflow_name}' doesn't match pattern: {name_pattern}",
                rule_id="NAMING_001"
            ))
        
        if len(workflow_name) > max_length:
            issues.append(QualityIssue(
                issue_id=str(uuid.uuid4()),
                gate_type=QualityGateType.NAMING_CONVENTION,
                severity=QualitySeverity.MEDIUM,
                message=f"Workflow name too long: {len(workflow_name)} > {max_length}",
                rule_id="NAMING_002"
            ))
        
        # Validate step IDs
        step_pattern = naming_rules.get("step_id_pattern", r"^[a-z][a-z0-9_]*$")
        steps = workflow_data.get("steps", [])
        
        for step in steps:
            if isinstance(step, dict):
                step_id = step.get("id", "")
                if step_id and not re.match(step_pattern, step_id):
                    issues.append(QualityIssue(
                        issue_id=str(uuid.uuid4()),
                        gate_type=QualityGateType.NAMING_CONVENTION,
                        severity=QualitySeverity.LOW,
                        message=f"Step ID '{step_id}' doesn't match pattern: {step_pattern}",
                        rule_id="NAMING_003"
                    ))
        
        return issues
    
    async def _complexity_analyzer(self, workflow_data: Dict[str, Any], metadata: Dict[str, Any], artifacts: Optional[Dict[str, Any]]) -> List[QualityIssue]:
        """Analyze workflow complexity"""
        issues = []
        
        steps = workflow_data.get("steps", [])
        
        # Calculate cyclomatic complexity (simplified)
        decision_steps = 0
        for step in steps:
            if isinstance(step, dict) and step.get("type") in ["decision", "ml_decision"]:
                decision_steps += 1
        
        complexity = decision_steps + 1  # Base complexity
        
        if complexity > 10:
            issues.append(QualityIssue(
                issue_id=str(uuid.uuid4()),
                gate_type=QualityGateType.COMPLEXITY_ANALYSIS,
                severity=QualitySeverity.MEDIUM,
                message=f"High cyclomatic complexity: {complexity}",
                rule_id="COMPLEX_001",
                suggestion="Consider simplifying decision logic"
            ))
        
        return issues
    
    async def _dependency_checker(self, workflow_data: Dict[str, Any], metadata: Dict[str, Any], artifacts: Optional[Dict[str, Any]]) -> List[QualityIssue]:
        """Check dependencies and external calls"""
        issues = []
        
        # Count external calls
        external_calls = 0
        steps = workflow_data.get("steps", [])
        
        for step in steps:
            if isinstance(step, dict):
                if step.get("type") == "agent_call":
                    external_calls += 1
                
                # Check for external URLs in params
                params = step.get("params", {})
                if isinstance(params, dict):
                    for value in params.values():
                        if isinstance(value, str) and ("http://" in value or "https://" in value):
                            external_calls += 1
        
        max_external = self.config.get("security_rules", {}).get("max_external_calls", 10)
        if external_calls > max_external:
            issues.append(QualityIssue(
                issue_id=str(uuid.uuid4()),
                gate_type=QualityGateType.DEPENDENCY_CHECK,
                severity=QualitySeverity.MEDIUM,
                message=f"Too many external calls: {external_calls} > {max_external}",
                rule_id="DEP_001",
                suggestion="Minimize external dependencies"
            ))
        
        return issues
    
    def _generate_summary(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Generate quality gate summary"""
        total_issues = sum(result.get("total_issues", 0) for result in results.values())
        critical_issues = sum(result.get("critical_count", 0) for result in results.values())
        high_issues = sum(result.get("high_count", 0) for result in results.values())
        
        passed_gates = sum(1 for result in results.values() if result.get("status") == "passed")
        failed_gates = sum(1 for result in results.values() if result.get("status") == "failed")
        
        return {
            "total_gates": len(results),
            "passed_gates": passed_gates,
            "failed_gates": failed_gates,
            "total_issues": total_issues,
            "critical_issues": critical_issues,
            "high_issues": high_issues,
            "quality_score": max(0, 100 - (critical_issues * 20) - (high_issues * 10))
        }
    
    def _generate_recommendations(self, results: Dict[str, Any]) -> List[str]:
        """Generate quality improvement recommendations"""
        recommendations = []
        
        for gate_name, result in results.items():
            if result.get("status") == "failed":
                critical_count = result.get("critical_count", 0)
                high_count = result.get("high_count", 0)
                
                if critical_count > 0:
                    recommendations.append(f"Fix {critical_count} critical issues in {gate_name}")
                if high_count > 0:
                    recommendations.append(f"Address {high_count} high-priority issues in {gate_name}")
        
        if not recommendations:
            recommendations.append("All quality gates passed - workflow is ready for publication")
        
        return recommendations


# Quality gate configuration templates
QUALITY_GATE_CONFIGS = {
    "strict": {
        "fail_on_critical": True,
        "fail_on_high": True,
        "max_issues_per_gate": 50,
        "gates": {gate.value: {"enabled": True} for gate in QualityGateType}
    },
    "standard": {
        "fail_on_critical": True,
        "fail_on_high": False,
        "max_issues_per_gate": 100,
        "gates": {
            "syntax_validation": {"enabled": True},
            "schema_compliance": {"enabled": True},
            "security_scan": {"enabled": True},
            "naming_convention": {"enabled": True},
            "documentation_check": {"enabled": False}
        }
    },
    "permissive": {
        "fail_on_critical": False,
        "fail_on_high": False,
        "max_issues_per_gate": 200,
        "gates": {
            "syntax_validation": {"enabled": True},
            "schema_compliance": {"enabled": True},
            "security_scan": {"enabled": False}
        }
    }
}
