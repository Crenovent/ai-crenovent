"""
SaaS Compiler & Runtime Enhancements
Policy injection and enforcement for SaaS RBA workflows
"""

from typing import Dict, List, Any, Optional, Union, Callable
from datetime import datetime, timedelta
from enum import Enum
from pydantic import BaseModel, Field
from abc import ABC, abstractmethod
import json
import logging
import asyncio
from contextlib import asynccontextmanager

from .parameter_packs import SaaSParameterPackManager, SaaSParameterCategory
from .schemas import (
    SaaSWorkflowTrace, SaaSAuditTrail, SaaSEvidencePack, 
    SaaSRuleViolation, SaaSOverrideRecord, SaaSSchemaFactory
)

logger = logging.getLogger(__name__)

class PolicyType(Enum):
    COMPLIANCE = "compliance"
    GOVERNANCE = "governance"
    SECURITY = "security"
    PERFORMANCE = "performance"
    DATA_QUALITY = "data_quality"

class PolicyEnforcementLevel(Enum):
    ADVISORY = "advisory"      # Log warning, continue execution
    BLOCKING = "blocking"      # Stop execution if violated
    OVERRIDE = "override"      # Allow override with approval

class SaaSPolicy(BaseModel):
    """Base policy model for SaaS workflows"""
    policy_id: str
    name: str
    description: str
    policy_type: PolicyType
    enforcement_level: PolicyEnforcementLevel
    applicable_workflows: List[str] = []  # Workflow IDs or patterns
    applicable_steps: List[str] = []      # Step types or IDs
    conditions: Dict[str, Any] = {}
    actions: Dict[str, Any] = {}
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
    is_active: bool = True
    
    class Config:
        use_enum_values = True

class SaaSCompliancePolicy(SaaSPolicy):
    """Compliance-specific policy for SaaS workflows"""
    compliance_framework: str  # SOX, GDPR, DSAR, etc.
    regulation_reference: str
    audit_required: bool = True
    evidence_collection_required: bool = True
    retention_days: int = 2555  # 7 years default for SOX
    
    def __init__(self, **data):
        data['policy_type'] = PolicyType.COMPLIANCE
        super().__init__(**data)

class SaaSGovernancePolicy(SaaSPolicy):
    """Governance-specific policy for SaaS workflows"""
    approval_required: bool = False
    approval_chain: List[str] = []
    maker_checker_required: bool = False
    override_allowed: bool = True
    escalation_rules: Dict[str, Any] = {}
    
    def __init__(self, **data):
        data['policy_type'] = PolicyType.GOVERNANCE
        super().__init__(**data)

class PolicyEvaluationResult(BaseModel):
    """Result of policy evaluation"""
    policy_id: str
    policy_name: str
    is_compliant: bool
    violations: List[SaaSRuleViolation] = []
    warnings: List[str] = []
    required_actions: List[str] = []
    evidence_collected: Dict[str, Any] = {}
    evaluation_time_ms: int
    
class PolicyInjector:
    """Injects policies into workflow compilation"""
    
    def __init__(self):
        self.policies: Dict[str, SaaSPolicy] = {}
        self.parameter_manager = SaaSParameterPackManager()
        self.schema_factory = SaaSSchemaFactory()
        self._load_default_policies()
    
    def _load_default_policies(self):
        """Load default SaaS policies"""
        # SOX Compliance Policy
        sox_policy = SaaSCompliancePolicy(
            policy_id="saas_sox_compliance",
            name="SOX Compliance for Financial Controls",
            description="Ensures SOX compliance for financial data and controls",
            compliance_framework="SOX",
            regulation_reference="Section 404",
            enforcement_level=PolicyEnforcementLevel.BLOCKING,
            applicable_workflows=["saas_pipeline_hygiene_v1.0", "saas_forecast_accuracy_v1.0"],
            conditions={
                "financial_impact_threshold": 10000,
                "requires_audit_trail": True,
                "immutable_logs_required": True
            },
            actions={
                "collect_evidence": True,
                "create_audit_trail": True,
                "notify_compliance": True
            }
        )
        self.policies[sox_policy.policy_id] = sox_policy
        
        # GDPR Data Privacy Policy
        gdpr_policy = SaaSCompliancePolicy(
            policy_id="saas_gdpr_privacy",
            name="GDPR Data Privacy Protection",
            description="Ensures GDPR compliance for personal data processing",
            compliance_framework="GDPR",
            regulation_reference="Article 6, Article 25",
            enforcement_level=PolicyEnforcementLevel.BLOCKING,
            applicable_workflows=["saas_lead_scoring_v1.0"],
            conditions={
                "personal_data_involved": True,
                "consent_required": True,
                "data_minimization": True
            },
            actions={
                "verify_consent": True,
                "anonymize_data": True,
                "log_processing_activity": True
            }
        )
        self.policies[gdpr_policy.policy_id] = gdpr_policy
        
        # Pipeline Coverage Governance Policy
        pipeline_governance = SaaSGovernancePolicy(
            policy_id="saas_pipeline_coverage_governance",
            name="Pipeline Coverage Governance",
            description="Governance rules for pipeline coverage overrides",
            enforcement_level=PolicyEnforcementLevel.OVERRIDE,
            applicable_workflows=["saas_pipeline_hygiene_v1.0"],
            approval_required=True,
            approval_chain=["sales_manager", "sales_director", "cro"],
            maker_checker_required=True,
            conditions={
                "coverage_ratio_threshold": 2.0,
                "override_requires_justification": True
            },
            actions={
                "create_override_record": True,
                "notify_approvers": True,
                "escalate_if_critical": True
            }
        )
        self.policies[pipeline_governance.policy_id] = pipeline_governance
    
    def add_policy(self, policy: SaaSPolicy):
        """Add a new policy"""
        self.policies[policy.policy_id] = policy
        logger.info(f"Added policy: {policy.name} ({policy.policy_id})")
    
    def get_applicable_policies(self, workflow_id: str, step_type: str = None) -> List[SaaSPolicy]:
        """Get policies applicable to a workflow/step"""
        applicable_policies = []
        
        for policy in self.policies.values():
            if not policy.is_active:
                continue
            
            # Check workflow applicability
            workflow_matches = (
                not policy.applicable_workflows or
                workflow_id in policy.applicable_workflows or
                any(pattern in workflow_id for pattern in policy.applicable_workflows)
            )
            
            # Check step applicability
            step_matches = (
                not step_type or
                not policy.applicable_steps or
                step_type in policy.applicable_steps
            )
            
            if workflow_matches and step_matches:
                applicable_policies.append(policy)
        
        return applicable_policies
    
    def inject_policies_into_workflow(self, workflow_definition: Dict[str, Any]) -> Dict[str, Any]:
        """Inject policy enforcement into workflow definition"""
        enhanced_workflow = workflow_definition.copy()
        workflow_id = workflow_definition.get("workflow_id")
        
        # Get applicable policies
        applicable_policies = self.get_applicable_policies(workflow_id)
        
        # Add policy metadata
        enhanced_workflow["injected_policies"] = [
            {
                "policy_id": policy.policy_id,
                "policy_name": policy.name,
                "enforcement_level": policy.enforcement_level.value
            }
            for policy in applicable_policies
        ]
        
        # Enhance each step with policy checks
        enhanced_steps = []
        for step in workflow_definition.get("steps", []):
            enhanced_step = step.copy()
            step_policies = self.get_applicable_policies(workflow_id, step.get("type"))
            
            if step_policies:
                # Add pre-execution policy checks
                enhanced_step["pre_execution_policies"] = [
                    self._create_policy_check(policy, "pre")
                    for policy in step_policies
                ]
                
                # Add post-execution policy checks
                enhanced_step["post_execution_policies"] = [
                    self._create_policy_check(policy, "post")
                    for policy in step_policies
                ]
            
            enhanced_steps.append(enhanced_step)
        
        enhanced_workflow["steps"] = enhanced_steps
        return enhanced_workflow
    
    def _create_policy_check(self, policy: SaaSPolicy, execution_phase: str) -> Dict[str, Any]:
        """Create policy check configuration"""
        return {
            "policy_id": policy.policy_id,
            "policy_name": policy.name,
            "enforcement_level": policy.enforcement_level.value,
            "execution_phase": execution_phase,
            "conditions": policy.conditions,
            "actions": policy.actions
        }

class PolicyEnforcer:
    """Enforces policies during workflow runtime"""
    
    def __init__(self):
        self.policy_injector = PolicyInjector()
        self.schema_factory = SaaSSchemaFactory()
        self.violation_handlers = {
            PolicyEnforcementLevel.ADVISORY: self._handle_advisory_violation,
            PolicyEnforcementLevel.BLOCKING: self._handle_blocking_violation,
            PolicyEnforcementLevel.OVERRIDE: self._handle_override_violation
        }
    
    async def evaluate_policies(
        self,
        workflow_id: str,
        step_id: str,
        step_type: str,
        execution_context: Dict[str, Any],
        execution_phase: str = "pre"
    ) -> List[PolicyEvaluationResult]:
        """Evaluate all applicable policies"""
        start_time = datetime.utcnow()
        
        applicable_policies = self.policy_injector.get_applicable_policies(workflow_id, step_type)
        evaluation_results = []
        
        for policy in applicable_policies:
            result = await self._evaluate_single_policy(
                policy, execution_context, execution_phase
            )
            evaluation_results.append(result)
        
        # Log policy evaluation
        end_time = datetime.utcnow()
        evaluation_time_ms = int((end_time - start_time).total_seconds() * 1000)
        
        logger.info(f"Evaluated {len(applicable_policies)} policies in {evaluation_time_ms}ms")
        
        return evaluation_results
    
    async def _evaluate_single_policy(
        self,
        policy: SaaSPolicy,
        execution_context: Dict[str, Any],
        execution_phase: str
    ) -> PolicyEvaluationResult:
        """Evaluate a single policy"""
        start_time = datetime.utcnow()
        
        violations = []
        warnings = []
        required_actions = []
        evidence_collected = {}
        is_compliant = True
        
        try:
            # Evaluate policy conditions
            for condition_name, condition_value in policy.conditions.items():
                compliance_result = await self._evaluate_condition(
                    condition_name, condition_value, execution_context
                )
                
                if not compliance_result["is_compliant"]:
                    is_compliant = False
                    violation = SaaSRuleViolation(
                        rule_id=policy.policy_id,
                        rule_name=policy.name,
                        rule_description=policy.description,
                        violation_type=policy.policy_type.value,
                        severity=self._determine_severity(policy.enforcement_level),
                        expected_value=condition_value,
                        actual_value=compliance_result["actual_value"],
                        business_impact_description=compliance_result.get("impact", "Policy violation"),
                        risk_category=policy.policy_type.value
                    )
                    violations.append(violation)
            
            # Collect evidence if required
            if policy.policy_type == PolicyType.COMPLIANCE:
                evidence_collected = await self._collect_compliance_evidence(
                    policy, execution_context
                )
            
            # Determine required actions
            if not is_compliant:
                required_actions = await self._determine_required_actions(
                    policy, violations, execution_context
                )
            
        except Exception as e:
            logger.error(f"Error evaluating policy {policy.policy_id}: {e}")
            is_compliant = False
            warnings.append(f"Policy evaluation error: {e}")
        
        end_time = datetime.utcnow()
        evaluation_time_ms = int((end_time - start_time).total_seconds() * 1000)
        
        return PolicyEvaluationResult(
            policy_id=policy.policy_id,
            policy_name=policy.name,
            is_compliant=is_compliant,
            violations=violations,
            warnings=warnings,
            required_actions=required_actions,
            evidence_collected=evidence_collected,
            evaluation_time_ms=evaluation_time_ms
        )
    
    async def _evaluate_condition(
        self,
        condition_name: str,
        expected_value: Any,
        execution_context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Evaluate a specific policy condition"""
        
        # Common condition evaluations
        if condition_name == "financial_impact_threshold":
            actual_value = execution_context.get("financial_impact", 0)
            return {
                "is_compliant": actual_value <= expected_value,
                "actual_value": actual_value,
                "impact": f"Financial impact of {actual_value} exceeds threshold of {expected_value}"
            }
        
        elif condition_name == "coverage_ratio_threshold":
            actual_value = execution_context.get("pipeline_coverage_ratio", 0)
            return {
                "is_compliant": actual_value >= expected_value,
                "actual_value": actual_value,
                "impact": f"Pipeline coverage ratio of {actual_value} below minimum of {expected_value}"
            }
        
        elif condition_name == "personal_data_involved":
            has_personal_data = execution_context.get("contains_personal_data", False)
            return {
                "is_compliant": not expected_value or not has_personal_data,
                "actual_value": has_personal_data,
                "impact": "Personal data processing detected, GDPR compliance required"
            }
        
        elif condition_name == "requires_audit_trail":
            has_audit_trail = execution_context.get("audit_trail_enabled", False)
            return {
                "is_compliant": not expected_value or has_audit_trail,
                "actual_value": has_audit_trail,
                "impact": "Audit trail required but not enabled"
            }
        
        # Default evaluation
        actual_value = execution_context.get(condition_name)
        return {
            "is_compliant": actual_value == expected_value,
            "actual_value": actual_value,
            "impact": f"Condition {condition_name} not met"
        }
    
    async def _collect_compliance_evidence(
        self,
        policy: SaaSPolicy,
        execution_context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Collect compliance evidence"""
        evidence = {
            "policy_id": policy.policy_id,
            "compliance_framework": getattr(policy, "compliance_framework", "unknown"),
            "collected_at": datetime.utcnow().isoformat(),
            "execution_context_snapshot": execution_context.copy()
        }
        
        # Add framework-specific evidence
        if hasattr(policy, "compliance_framework"):
            if policy.compliance_framework == "SOX":
                evidence["sox_control_evidence"] = {
                    "control_tested": True,
                    "test_result": "pending",
                    "financial_data_accessed": execution_context.get("financial_data_accessed", [])
                }
            elif policy.compliance_framework == "GDPR":
                evidence["gdpr_processing_evidence"] = {
                    "processing_purpose": execution_context.get("processing_purpose", "unknown"),
                    "legal_basis": execution_context.get("legal_basis", "legitimate_interest"),
                    "data_subjects_affected": execution_context.get("data_subjects_count", 0)
                }
        
        return evidence
    
    async def _determine_required_actions(
        self,
        policy: SaaSPolicy,
        violations: List[SaaSRuleViolation],
        execution_context: Dict[str, Any]
    ) -> List[str]:
        """Determine required actions based on violations"""
        actions = []
        
        # Add policy-specific actions
        for action_name, action_config in policy.actions.items():
            if action_config:
                actions.append(action_name)
        
        # Add violation-specific actions
        for violation in violations:
            if violation.severity == "critical":
                actions.append("immediate_escalation")
            elif violation.severity == "high":
                actions.append("notify_compliance_team")
        
        return actions
    
    def _determine_severity(self, enforcement_level: PolicyEnforcementLevel) -> str:
        """Determine violation severity based on enforcement level"""
        severity_mapping = {
            PolicyEnforcementLevel.ADVISORY: "medium",
            PolicyEnforcementLevel.BLOCKING: "high",
            PolicyEnforcementLevel.OVERRIDE: "critical"
        }
        return severity_mapping.get(enforcement_level, "medium")
    
    async def handle_policy_violations(
        self,
        evaluation_results: List[PolicyEvaluationResult],
        execution_context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Handle policy violations based on enforcement levels"""
        handling_results = {
            "can_continue": True,
            "violations_handled": [],
            "overrides_required": [],
            "escalations_created": [],
            "evidence_collected": {}
        }
        
        for result in evaluation_results:
            if not result.is_compliant:
                policy = self.policy_injector.policies[result.policy_id]
                handler = self.violation_handlers[policy.enforcement_level]
                
                handling_result = await handler(result, policy, execution_context)
                
                # Aggregate results
                if not handling_result.get("can_continue", True):
                    handling_results["can_continue"] = False
                
                handling_results["violations_handled"].append({
                    "policy_id": result.policy_id,
                    "handling_result": handling_result
                })
        
        return handling_results
    
    async def _handle_advisory_violation(
        self,
        result: PolicyEvaluationResult,
        policy: SaaSPolicy,
        execution_context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Handle advisory-level policy violations"""
        logger.warning(f"Advisory policy violation: {policy.name}")
        
        return {
            "can_continue": True,
            "action_taken": "logged_warning",
            "notification_sent": False
        }
    
    async def _handle_blocking_violation(
        self,
        result: PolicyEvaluationResult,
        policy: SaaSPolicy,
        execution_context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Handle blocking-level policy violations"""
        logger.error(f"Blocking policy violation: {policy.name}")
        
        return {
            "can_continue": False,
            "action_taken": "execution_blocked",
            "reason": f"Policy violation: {policy.name}",
            "violations": [v.dict() for v in result.violations]
        }
    
    async def _handle_override_violation(
        self,
        result: PolicyEvaluationResult,
        policy: SaaSPolicy,
        execution_context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Handle override-level policy violations"""
        logger.warning(f"Override-required policy violation: {policy.name}")
        
        # Create override record
        override_record = SaaSOverrideRecord(
            workflow_execution_id=execution_context.get("execution_id"),
            rule_id=policy.policy_id,
            rule_name=policy.name,
            override_reason="Policy violation requires approval",
            override_justification="Automated override request due to policy violation",
            override_type="automatic",
            requested_by=execution_context.get("user_id", "system"),
            original_value="policy_compliant",
            override_value="policy_violation_override_requested",
            risk_assessment="medium",
            business_justification="Override required to continue workflow execution"
        )
        
        return {
            "can_continue": False,  # Require override approval
            "action_taken": "override_requested",
            "override_record": override_record.dict(),
            "approval_required": True,
            "approval_chain": getattr(policy, "approval_chain", [])
        }

class SaaSRuntimeEnhancer:
    """Runtime enhancements for SaaS workflow execution"""
    
    def __init__(self):
        self.policy_enforcer = PolicyEnforcer()
        self.schema_factory = SaaSSchemaFactory()
        self.active_executions: Dict[str, Dict[str, Any]] = {}
    
    @asynccontextmanager
    async def enhanced_execution_context(
        self,
        execution_id: str,
        workflow_id: str,
        tenant_id: str,
        user_id: str = None
    ):
        """Context manager for enhanced workflow execution"""
        
        # Initialize execution context
        execution_context = {
            "execution_id": execution_id,
            "workflow_id": workflow_id,
            "tenant_id": tenant_id,
            "user_id": user_id,
            "started_at": datetime.utcnow(),
            "audit_trail_enabled": True,
            "policy_enforcement_enabled": True
        }
        
        self.active_executions[execution_id] = execution_context
        
        try:
            yield execution_context
        finally:
            # Cleanup
            execution_context["ended_at"] = datetime.utcnow()
            execution_context["duration_ms"] = int(
                (execution_context["ended_at"] - execution_context["started_at"]).total_seconds() * 1000
            )
            
            # Create final audit trail
            await self._create_final_audit_trail(execution_context)
            
            # Remove from active executions
            self.active_executions.pop(execution_id, None)
    
    async def execute_step_with_policies(
        self,
        step_definition: Dict[str, Any],
        execution_context: Dict[str, Any],
        step_executor: Callable
    ) -> Dict[str, Any]:
        """Execute a workflow step with policy enforcement"""
        
        step_id = step_definition.get("id")
        step_type = step_definition.get("type")
        workflow_id = execution_context.get("workflow_id")
        
        logger.info(f"Executing step {step_id} with policy enforcement")
        
        # Pre-execution policy evaluation
        pre_evaluation_results = await self.policy_enforcer.evaluate_policies(
            workflow_id, step_id, step_type, execution_context, "pre"
        )
        
        # Handle pre-execution violations
        pre_violation_handling = await self.policy_enforcer.handle_policy_violations(
            pre_evaluation_results, execution_context
        )
        
        if not pre_violation_handling["can_continue"]:
            return {
                "success": False,
                "error": "Pre-execution policy violations prevent step execution",
                "policy_violations": pre_violation_handling
            }
        
        # Execute the actual step
        try:
            step_result = await step_executor(step_definition, execution_context)
            
            # Update execution context with step results
            execution_context.update({
                f"step_{step_id}_result": step_result,
                f"step_{step_id}_completed_at": datetime.utcnow()
            })
            
        except Exception as e:
            logger.error(f"Step execution failed: {e}")
            return {
                "success": False,
                "error": str(e),
                "step_id": step_id
            }
        
        # Post-execution policy evaluation
        post_evaluation_results = await self.policy_enforcer.evaluate_policies(
            workflow_id, step_id, step_type, execution_context, "post"
        )
        
        # Handle post-execution violations
        post_violation_handling = await self.policy_enforcer.handle_policy_violations(
            post_evaluation_results, execution_context
        )
        
        # Create step trace
        step_trace = self._create_step_trace(
            step_definition, execution_context, step_result,
            pre_evaluation_results + post_evaluation_results
        )
        
        return {
            "success": True,
            "step_result": step_result,
            "step_trace": step_trace,
            "policy_evaluations": {
                "pre_execution": pre_evaluation_results,
                "post_execution": post_evaluation_results
            },
            "violation_handling": {
                "pre_execution": pre_violation_handling,
                "post_execution": post_violation_handling
            }
        }
    
    def _create_step_trace(
        self,
        step_definition: Dict[str, Any],
        execution_context: Dict[str, Any],
        step_result: Dict[str, Any],
        policy_evaluations: List[PolicyEvaluationResult]
    ) -> SaaSWorkflowTrace:
        """Create trace record for step execution"""
        
        # Determine workflow type from workflow_id
        workflow_id = execution_context.get("workflow_id", "")
        if "pipeline_hygiene" in workflow_id:
            workflow_type = "pipeline_hygiene"
        elif "forecast_accuracy" in workflow_id:
            workflow_type = "forecast_accuracy"
        elif "lead_scoring" in workflow_id:
            workflow_type = "lead_scoring"
        else:
            workflow_type = "unknown"
        
        trace = self.schema_factory.create_trace(
            tenant_id=execution_context.get("tenant_id"),
            workflow_type=workflow_type,
            execution_id=execution_context.get("execution_id"),
            workflow_id=execution_context.get("workflow_id"),
            step_id=step_definition.get("id"),
            event_type="step_completed",
            workflow_name=step_definition.get("name", "Unknown"),
            step_name=step_definition.get("name", "Unknown"),
            step_type=step_definition.get("type", "unknown"),
            execution_context=execution_context.copy(),
            input_data=step_definition.get("config", {}),
            output_data=step_result,
            execution_time_ms=step_result.get("execution_time_ms", 0)
        )
        
        # Add policy evaluation results to trace
        trace.compliance_frameworks = []
        for evaluation in policy_evaluations:
            if not evaluation.is_compliant:
                trace.errors.extend([v.dict() for v in evaluation.violations])
        
        return trace
    
    async def _create_final_audit_trail(self, execution_context: Dict[str, Any]):
        """Create final audit trail for workflow execution"""
        try:
            audit_trail = self.schema_factory.create_audit_trail(
                tenant_id=execution_context.get("tenant_id"),
                workflow_type="pipeline_hygiene",  # Default, should be determined dynamically
                workflow_execution_id=execution_context.get("execution_id"),
                audit_type="execution",
                execution_summary=execution_context,
                total_execution_time_ms=execution_context.get("duration_ms", 0)
            )
            
            logger.info(f"Created audit trail: {audit_trail.audit_id}")
            
        except Exception as e:
            logger.error(f"Failed to create audit trail: {e}")

# Example usage and testing
if __name__ == "__main__":
    import asyncio
    
    async def test_policy_enforcement():
        """Test policy enforcement functionality"""
        
        # Initialize components
        injector = PolicyInjector()
        enforcer = PolicyEnforcer()
        runtime_enhancer = SaaSRuntimeEnhancer()
        
        # Test workflow definition
        workflow_definition = {
            "workflow_id": "saas_pipeline_hygiene_v1.0",
            "name": "Pipeline Hygiene Test",
            "steps": [
                {
                    "id": "calculate_metrics",
                    "type": "calculation",
                    "name": "Calculate Pipeline Metrics"
                }
            ]
        }
        
        # Inject policies
        enhanced_workflow = injector.inject_policies_into_workflow(workflow_definition)
        print(f"Injected policies: {len(enhanced_workflow.get('injected_policies', []))}")
        
        # Test policy evaluation
        execution_context = {
            "execution_id": "test_exec_123",
            "workflow_id": "saas_pipeline_hygiene_v1.0",
            "tenant_id": "tenant_123",
            "pipeline_coverage_ratio": 1.5,  # Below threshold
            "financial_impact": 15000        # Above threshold
        }
        
        evaluation_results = await enforcer.evaluate_policies(
            "saas_pipeline_hygiene_v1.0",
            "calculate_metrics",
            "calculation",
            execution_context
        )
        
        print(f"Policy evaluations: {len(evaluation_results)}")
        for result in evaluation_results:
            print(f"  - {result.policy_name}: {'COMPLIANT' if result.is_compliant else 'VIOLATION'}")
            if result.violations:
                print(f"    Violations: {len(result.violations)}")
    
    # Run test
    asyncio.run(test_policy_enforcement())
