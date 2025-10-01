"""
DSL Workflow Validator
=====================

Implements Task 6.2-T03: Static analyzer for governance and schema validation
Validates workflows for compliance, governance, and structural integrity.
"""

import logging
from typing import Dict, List, Any, Optional, Set
from dataclasses import dataclass
from enum import Enum

from .parser import DSLWorkflowAST, DSLStep, StepType

logger = logging.getLogger(__name__)

class ValidationLevel(Enum):
    """Validation severity levels"""
    ERROR = "error"
    WARNING = "warning" 
    INFO = "info"

@dataclass
class ValidationIssue:
    """Single validation issue"""
    level: ValidationLevel
    message: str
    step_id: Optional[str] = None
    field: Optional[str] = None
    suggestion: Optional[str] = None

@dataclass 
class ValidationResult:
    """Complete validation result"""
    is_valid: bool
    issues: List[ValidationIssue]
    warnings_count: int = 0
    errors_count: int = 0
    
    def __post_init__(self):
        self.warnings_count = sum(1 for issue in self.issues if issue.level == ValidationLevel.WARNING)
        self.errors_count = sum(1 for issue in self.issues if issue.level == ValidationLevel.ERROR)
        self.is_valid = self.errors_count == 0

class WorkflowValidator:
    """
    Static validator for DSL workflows
    
    Task 6.2-T03: Implement static analyzer (lint for schema, governance fields)
    """
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
        # Required fields for different step types
        self.required_fields = {
            StepType.QUERY: ['params'],
            StepType.DECISION: ['params', 'conditions'],
            StepType.ACTION: ['params'],
            StepType.NOTIFY: ['params'],
            StepType.GOVERNANCE: ['params'],
            StepType.AGENT_CALL: ['params'],
            StepType.ML_DECISION: ['params']
        }
        
        # Required governance fields for compliance
        self.governance_requirements = {
            'evidence_required': bool,
            'policy_packs': list,
            'override_allowed': bool
        }
    
    def validate_workflow(self, ast: DSLWorkflowAST) -> ValidationResult:
        """
        Validate complete workflow AST
        
        Returns:
            ValidationResult with all issues found
        """
        issues = []
        
        try:
            # Basic structure validation
            issues.extend(self._validate_structure(ast))
            
            # Step-by-step validation
            issues.extend(self._validate_steps(ast))
            
            # Governance validation (Task 6.2-T36: Build policy guardrails)
            issues.extend(self._validate_governance(ast))
            
            # Flow validation (check for cycles, unreachable steps)
            issues.extend(self._validate_workflow_flow(ast))
            
            # SaaS-specific validation
            issues.extend(self._validate_saas_compliance(ast))
            
            result = ValidationResult(is_valid=True, issues=issues)
            
            if result.errors_count > 0:
                self.logger.warning(f"❌ Workflow validation failed: {result.errors_count} errors, {result.warnings_count} warnings")
            else:
                self.logger.info(f"✅ Workflow validation passed: {result.warnings_count} warnings")
                
            return result
            
        except Exception as e:
            self.logger.error(f"❌ Validation failed with exception: {str(e)}")
            issues.append(ValidationIssue(
                level=ValidationLevel.ERROR,
                message=f"Validation exception: {str(e)}"
            ))
            return ValidationResult(is_valid=False, issues=issues)
    
    def _validate_structure(self, ast: DSLWorkflowAST) -> List[ValidationIssue]:
        """Validate basic workflow structure"""
        issues = []
        
        # Check required fields
        if not ast.workflow_id:
            issues.append(ValidationIssue(
                level=ValidationLevel.ERROR,
                message="Workflow ID is required",
                field="workflow_id"
            ))
            
        if not ast.name:
            issues.append(ValidationIssue(
                level=ValidationLevel.ERROR,
                message="Workflow name is required", 
                field="name"
            ))
        
        if not ast.steps:
            issues.append(ValidationIssue(
                level=ValidationLevel.ERROR,
                message="Workflow must contain at least one step",
                field="steps"
            ))
        
        # Check for duplicate step IDs
        step_ids = [step.step_id for step in ast.steps]
        duplicates = set([x for x in step_ids if step_ids.count(x) > 1])
        for dup_id in duplicates:
            issues.append(ValidationIssue(
                level=ValidationLevel.ERROR,
                message=f"Duplicate step ID: {dup_id}",
                field="steps"
            ))
        
        return issues
    
    def _validate_steps(self, ast: DSLWorkflowAST) -> List[ValidationIssue]:
        """Validate individual steps"""
        issues = []
        
        for step in ast.steps:
            # Check required fields for step type
            required = self.required_fields.get(step.type, [])
            for field in required:
                if not hasattr(step, field) or getattr(step, field) is None:
                    issues.append(ValidationIssue(
                        level=ValidationLevel.ERROR,
                        message=f"Required field '{field}' missing for {step.type.value} step",
                        step_id=step.step_id,
                        field=field
                    ))
            
            # Validate step-specific requirements
            issues.extend(self._validate_step_specific(step))
            
            # Validate next_steps references
            issues.extend(self._validate_step_references(step, ast))
        
        return issues
    
    def _validate_step_specific(self, step: DSLStep) -> List[ValidationIssue]:
        """Validate step-specific requirements"""
        issues = []
        
        if step.type == StepType.QUERY:
            # Query steps should have operator or source
            params = step.params or {}
            if not params.get('operator') and not params.get('source'):
                issues.append(ValidationIssue(
                    level=ValidationLevel.ERROR,
                    message="Query step must specify 'operator' or 'source'",
                    step_id=step.step_id,
                    field="params"
                ))
        
        elif step.type == StepType.DECISION:
            # Decision steps should have conditions
            if not step.conditions:
                issues.append(ValidationIssue(
                    level=ValidationLevel.ERROR,
                    message="Decision step must have conditions",
                    step_id=step.step_id,
                    field="conditions"
                ))
        
        elif step.type == StepType.NOTIFY:
            # Notify steps should have channels and message
            params = step.params or {}
            if not params.get('channels'):
                issues.append(ValidationIssue(
                    level=ValidationLevel.WARNING,
                    message="Notify step should specify channels",
                    step_id=step.step_id,
                    field="params.channels"
                ))
                
        elif step.type == StepType.GOVERNANCE:
            # Governance steps should have policy references
            params = step.params or {}
            if not params.get('policy_id') and not params.get('action'):
                issues.append(ValidationIssue(
                    level=ValidationLevel.WARNING,
                    message="Governance step should specify policy_id or action",
                    step_id=step.step_id,
                    field="params"
                ))
        
        return issues
    
    def _validate_step_references(self, step: DSLStep, ast: DSLWorkflowAST) -> List[ValidationIssue]:
        """Validate step references (next_steps)"""
        issues = []
        
        all_step_ids = {s.step_id for s in ast.steps}
        
        for next_step_id in step.next_steps:
            if next_step_id not in all_step_ids:
                issues.append(ValidationIssue(
                    level=ValidationLevel.ERROR,
                    message=f"Reference to non-existent step: {next_step_id}",
                    step_id=step.step_id,
                    field="next_steps"
                ))
        
        # Check for self-references (infinite loops)
        if step.step_id in step.next_steps:
            issues.append(ValidationIssue(
                level=ValidationLevel.ERROR,
                message=f"Step cannot reference itself: {step.step_id}",
                step_id=step.step_id,
                field="next_steps"
            ))
        
        return issues
    
    def _validate_governance(self, ast: DSLWorkflowAST) -> List[ValidationIssue]:
        """
        Validate governance requirements
        Task 6.2-T36: Build policy guardrails at authoring time
        """
        issues = []
        
        governance = ast.governance or {}
        
        # Check required governance fields
        if not governance.get('policy_packs'):
            issues.append(ValidationIssue(
                level=ValidationLevel.WARNING,
                message="Workflow should specify policy packs for compliance",
                field="governance.policy_packs",
                suggestion="Add relevant policy packs like 'saas_compliance', 'sox_audit'"
            ))
        
        if governance.get('evidence_required') is None:
            issues.append(ValidationIssue(
                level=ValidationLevel.WARNING,
                message="Workflow should explicitly set evidence_required",
                field="governance.evidence_required",
                suggestion="Set evidence_required: true for audit compliance"
            ))
        
        # Validate policy pack references
        policy_packs = governance.get('policy_packs', [])
        valid_policy_packs = [
            'saas_compliance', 'sox_audit', 'gdpr_compliance', 
            'forecast_governance', 'pipeline_hygiene', 'revenue_ops'
        ]
        
        for policy_pack in policy_packs:
            if policy_pack not in valid_policy_packs:
                issues.append(ValidationIssue(
                    level=ValidationLevel.WARNING,
                    message=f"Unknown policy pack: {policy_pack}",
                    field="governance.policy_packs",
                    suggestion=f"Use one of: {', '.join(valid_policy_packs)}"
                ))
        
        return issues
    
    def _validate_workflow_flow(self, ast: DSLWorkflowAST) -> List[ValidationIssue]:
        """Validate workflow flow for cycles and unreachable steps"""
        issues = []
        
        if not ast.steps:
            return issues
        
        # Build adjacency graph
        graph = {}
        for step in ast.steps:
            graph[step.step_id] = step.next_steps
        
        # Check for cycles using DFS
        visited = set()
        rec_stack = set()
        
        def has_cycle(node):
            if node in rec_stack:
                return True
            if node in visited:
                return False
            
            visited.add(node)
            rec_stack.add(node)
            
            for neighbor in graph.get(node, []):
                if has_cycle(neighbor):
                    return True
            
            rec_stack.remove(node)
            return False
        
        for step_id in graph:
            if step_id not in visited:
                if has_cycle(step_id):
                    issues.append(ValidationIssue(
                        level=ValidationLevel.ERROR,
                        message=f"Workflow contains cycle involving step: {step_id}",
                        step_id=step_id,
                        field="next_steps"
                    ))
                    break
        
        # Check for unreachable steps (except first step)
        if ast.steps:
            reachable = set()
            start_step = ast.steps[0].step_id
            
            def dfs_reachable(node):
                if node in reachable:
                    return
                reachable.add(node)
                for neighbor in graph.get(node, []):
                    dfs_reachable(neighbor)
            
            dfs_reachable(start_step)
            
            for step in ast.steps[1:]:  # Skip first step
                if step.step_id not in reachable:
                    issues.append(ValidationIssue(
                        level=ValidationLevel.WARNING,
                        message=f"Step may be unreachable: {step.step_id}",
                        step_id=step.step_id,
                        suggestion="Ensure step is referenced by previous steps"
                    ))
        
        return issues
    
    def _validate_saas_compliance(self, ast: DSLWorkflowAST) -> List[ValidationIssue]:
        """Validate SaaS-specific compliance requirements"""
        issues = []
        
        # Check for SaaS-specific governance requirements
        governance = ast.governance or {}
        policy_packs = governance.get('policy_packs', [])
        
        # SaaS workflows should include compliance policies
        saas_policies = ['saas_compliance', 'sox_audit', 'pipeline_hygiene']
        has_saas_policy = any(policy in policy_packs for policy in saas_policies)
        
        if not has_saas_policy:
            issues.append(ValidationIssue(
                level=ValidationLevel.INFO,
                message="SaaS workflow should include relevant compliance policies",
                field="governance.policy_packs",
                suggestion=f"Consider adding: {', '.join(saas_policies)}"
            ))
        
        # Check for evidence requirements in SaaS workflows
        if governance.get('evidence_required') is False:
            issues.append(ValidationIssue(
                level=ValidationLevel.WARNING,
                message="SaaS workflows typically require evidence for compliance",
                field="governance.evidence_required",
                suggestion="Set evidence_required: true for SOX compliance"
            ))
        
        return issues
