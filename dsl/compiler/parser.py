"""
DSL Compiler - Parser Component
==============================

Implements Task 6.2-T01, T02, T03:
- Define DSL grammar for RBA workflows
- Build parser to translate DSL into AST
- Implement static analyzer for validation

Parses YAML/JSON workflows into Abstract Syntax Tree for execution.
"""

import yaml
import json
import logging
import hashlib
from typing import Dict, List, Any, Optional, Union
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime

logger = logging.getLogger(__name__)

class StepType(Enum):
    """DSL Step Types - Task 6.2-T01 with SaaS extensions"""
    # Base DSL primitives
    QUERY = "query"
    DECISION = "decision" 
    ACTION = "action"
    NOTIFY = "notify"
    GOVERNANCE = "governance"
    AGENT_CALL = "agent_call"
    ML_DECISION = "ml_decision"
    
    # SaaS-specific primitives (Task 7.1.1)
    SUBSCRIPTION = "subscription"          # Subscription lifecycle operations
    BILLING = "billing"                   # Billing and revenue operations
    CUSTOMER_SUCCESS = "customer_success" # Customer health and success metrics
    CHURN_ANALYSIS = "churn_analysis"     # Churn prediction and prevention
    REVENUE_RECOGNITION = "revenue_recognition" # Revenue recognition workflows
    USAGE_METERING = "usage_metering"     # Usage-based billing metering

class WorkflowStatus(Enum):
    """Workflow execution status"""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"

@dataclass
class DSLStep:
    """Single step in DSL workflow with SaaS extensions"""
    step_id: str
    name: str
    type: StepType
    params: Dict[str, Any] = field(default_factory=dict)
    conditions: Optional[Dict[str, Any]] = None
    next_steps: List[str] = field(default_factory=list)
    # SaaS-specific extensions
    saas_params: Optional[Dict[str, Any]] = None  # SaaS-specific parameters
    tenant_context: Optional[Dict[str, Any]] = None  # Multi-tenant context
    
    # Mandatory governance fields (Task 7.1.3)
    tenant_id: Optional[int] = None  # Required for tenant isolation
    region_id: Optional[str] = None  # Required for residency enforcement
    sla_tier: Optional[str] = None   # Required for SLA-aware execution
    policy_pack_id: Optional[str] = None  # Required for policy enforcement
    governance: Dict[str, Any] = field(default_factory=dict)
    retry_config: Dict[str, Any] = field(default_factory=dict)

@dataclass
class DSLWorkflowAST:
    """Abstract Syntax Tree representation of DSL workflow"""
    workflow_id: str
    name: str
    description: str
    version: str = "1.0.0"
    steps: List[DSLStep] = field(default_factory=list)
    governance: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    plan_hash: Optional[str] = None
    created_at: datetime = field(default_factory=datetime.utcnow)

class DSLCompiler:
    """
    DSL Compiler - Parses workflows into executable AST
    
    Task 6.2-T02: Build parser to translate DSL into AST
    Task 6.2-T03: Implement static analyzer
    Task 7.1.3: Configure mandatory metadata fields
    Task 7.1.6: Create policy-as-tests integration
    Task 7.1.12: Automate DSL linting
    """
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.supported_formats = ['yaml', 'yml', 'json']
        
        # Policy and linting configuration (Task 7.1.6, 7.1.12)
        self.policy_validation_enabled = True
        self.linting_enabled = True
        self.mandatory_governance_fields = ['tenant_id', 'region_id', 'sla_tier', 'policy_pack_id']
        self.linting_rules = self._load_linting_rules()
        
    def compile_workflow(self, workflow_definition: Union[str, Dict], format: str = 'yaml') -> DSLWorkflowAST:
        """
        Compile workflow definition into AST
        
        Args:
            workflow_definition: YAML/JSON string or dict
            format: 'yaml', 'yml', or 'json'
            
        Returns:
            DSLWorkflowAST: Parsed and validated workflow
        """
        try:
            # Parse the workflow definition
            if isinstance(workflow_definition, str):
                workflow_dict = self._parse_string(workflow_definition, format)
            else:
                workflow_dict = workflow_definition
                
            # Create AST from parsed definition
            ast = self._create_ast(workflow_dict)
            
            # Validate mandatory governance fields (Task 7.1.3)
            self._validate_governance_fields(ast)
            
            # Run DSL linting (Task 7.1.12)
            if self.linting_enabled:
                lint_results = self._lint_workflow(ast)
                if lint_results['errors']:
                    raise CompilationError(f"Linting failed: {lint_results['errors']}")
            
            # Policy-as-tests validation (Task 7.1.6)
            if self.policy_validation_enabled:
                policy_results = self._validate_policy_compliance(ast)
                if not policy_results['compliant']:
                    raise CompilationError(f"Policy validation failed: {policy_results['violations']}")
            
            # Generate plan hash for audit trail (Task 6.2-T04)
            ast.plan_hash = self._generate_plan_hash(ast)
            
            self.logger.info(f"✅ Compiled workflow: {ast.name} (hash: {ast.plan_hash[:8]})")
            return ast
            
        except Exception as e:
            self.logger.error(f"❌ Failed to compile workflow: {str(e)}")
            raise CompilationError(f"Workflow compilation failed: {str(e)}")
    
    def _parse_string(self, definition: str, format: str) -> Dict[str, Any]:
        """Parse string definition into dictionary"""
        try:
            if format.lower() in ['yaml', 'yml']:
                return yaml.safe_load(definition)
            elif format.lower() == 'json':
                return json.loads(definition)
            else:
                raise ValueError(f"Unsupported format: {format}")
        except Exception as e:
            raise CompilationError(f"Failed to parse {format}: {str(e)}")
    
    def _create_ast(self, workflow_dict: Dict[str, Any]) -> DSLWorkflowAST:
        """Create AST from workflow dictionary"""
        # Extract workflow metadata
        workflow_id = workflow_dict.get('workflow_id', workflow_dict.get('id', 'unknown'))
        name = workflow_dict.get('name', 'Unnamed Workflow')
        description = workflow_dict.get('description', '')
        version = workflow_dict.get('version', '1.0.0')
        
        # Parse steps
        steps = []
        steps_data = workflow_dict.get('steps', [])
        
        for i, step_data in enumerate(steps_data):
            step = self._parse_step(step_data, i)
            steps.append(step)
        
        # Extract governance configuration
        governance = workflow_dict.get('governance', {})
        metadata = workflow_dict.get('metadata', {})
        
        return DSLWorkflowAST(
            workflow_id=workflow_id,
            name=name,
            description=description,
            version=version,
            steps=steps,
            governance=governance,
            metadata=metadata
        )
    
    def _parse_step(self, step_data: Dict[str, Any], index: int) -> DSLStep:
        """Parse individual step from dictionary"""
        step_id = step_data.get('id', f"step_{index}")
        name = step_data.get('name', f"Step {index + 1}")
        
        # Parse step type
        step_type_str = step_data.get('type', 'action')
        try:
            step_type = StepType(step_type_str)
        except ValueError:
            raise CompilationError(f"Invalid step type: {step_type_str}")
        
        # Extract other properties
        params = step_data.get('params', {})
        conditions = step_data.get('conditions')
        next_steps = step_data.get('next_steps', step_data.get('next', []))
        governance = step_data.get('governance', {})
        retry_config = step_data.get('retry', {})
        
        # Ensure next_steps is a list
        if isinstance(next_steps, str):
            next_steps = [next_steps]
        elif next_steps is None:
            next_steps = []
        
        return DSLStep(
            step_id=step_id,
            name=name,
            type=step_type,
            params=params,
            conditions=conditions,
            next_steps=next_steps,
            governance=governance,
            retry_config=retry_config
        )
    
    def _generate_plan_hash(self, ast: DSLWorkflowAST) -> str:
        """
        Generate deterministic hash of execution plan
        Task 6.2-T04: Create workflow planner with plan hash
        """
        # Create a deterministic representation
        plan_data = {
            'workflow_id': ast.workflow_id,
            'version': ast.version,
            'steps': [
                {
                    'id': step.step_id,
                    'type': step.type.value,
                    'params': step.params,
                    'next_steps': sorted(step.next_steps)
                }
                for step in ast.steps
            ],
            'governance': ast.governance
        }
        
        # Generate SHA256 hash
        plan_json = json.dumps(plan_data, sort_keys=True, separators=(',', ':'))
        return hashlib.sha256(plan_json.encode()).hexdigest()
    
    def _load_linting_rules(self) -> Dict[str, Any]:
        """Load DSL linting rules - Task 7.1.12"""
        return {
            'syntax_rules': {
                'required_fields': ['workflow_id', 'name', 'steps'],
                'step_required_fields': ['id', 'name', 'type'],
                'valid_step_types': [t.value for t in StepType]
            },
            'governance_rules': {
                'mandatory_governance_fields': self.mandatory_governance_fields,
                'policy_pack_required': True,
                'tenant_isolation_required': True
            },
            'saas_rules': {
                'subscription_workflows_require_billing': True,
                'revenue_workflows_require_sox_compliance': True,
                'customer_data_requires_gdpr_compliance': True
            }
        }
    
    def _validate_governance_fields(self, ast: DSLWorkflowAST) -> None:
        """Validate mandatory governance fields - Task 7.1.3"""
        missing_fields = []
        
        # Check workflow-level governance fields
        for field in self.mandatory_governance_fields:
            if field not in ast.governance or not ast.governance[field]:
                missing_fields.append(f"workflow.governance.{field}")
        
        if missing_fields:
            raise CompilationError(f"Missing mandatory governance fields: {missing_fields}")
    
    def _lint_workflow(self, ast: DSLWorkflowAST) -> Dict[str, Any]:
        """Run DSL linting checks - Task 7.1.12"""
        errors = []
        warnings = []
        
        # Syntax validation
        if not ast.workflow_id:
            errors.append("Missing workflow_id")
        if not ast.name:
            errors.append("Missing workflow name")
        if not ast.steps:
            errors.append("Workflow must have at least one step")
        
        # Governance linting
        if not ast.governance.get('policy_pack_id'):
            errors.append("Workflow must specify a policy_pack_id")
        
        return {
            'errors': errors,
            'warnings': warnings,
            'compliant': len(errors) == 0
        }
    
    def _validate_policy_compliance(self, ast: DSLWorkflowAST) -> Dict[str, Any]:
        """Validate policy compliance - Task 7.1.6"""
        violations = []
        
        # Check SaaS-specific policy requirements
        has_revenue_steps = any(step.type in [StepType.BILLING, StepType.REVENUE_RECOGNITION] for step in ast.steps)
        has_customer_data_steps = any(step.type in [StepType.CUSTOMER_SUCCESS, StepType.CHURN_ANALYSIS] for step in ast.steps)
        
        if has_revenue_steps and 'SOX_SAAS' not in ast.governance.get('compliance_requirements', []):
            violations.append("Revenue workflows require SOX_SAAS compliance")
        
        if has_customer_data_steps and 'GDPR_SAAS' not in ast.governance.get('compliance_requirements', []):
            violations.append("Customer data workflows require GDPR_SAAS compliance")
        
        # Check tenant isolation
        if not ast.governance.get('tenant_id'):
            violations.append("All workflows must specify tenant_id for isolation")
        
        return {
            'compliant': len(violations) == 0,
            'violations': violations
        }

class CompilationError(Exception):
    """Exception raised during workflow compilation"""
    pass

# Example workflow definitions for testing
SAMPLE_PIPELINE_HYGIENE_WORKFLOW = """
workflow_id: "pipeline_hygiene_workflow"
name: "SaaS Pipeline Hygiene Check"
description: "Automated pipeline health monitoring with quota compliance"
version: "1.0.0"

governance:
  policy_packs: ["saas_compliance", "sox_audit"]
  evidence_required: true
  override_allowed: false

steps:
  - id: "query_opportunities"
    name: "Query Stale Opportunities"
    type: "query"
    params:
      operator: "pipeline_hygiene"
      config:
        stale_days_threshold: 15
        required_fields: ["amount", "close_date", "stage", "probability"]
        min_pipeline_coverage: 3.0
    governance:
      evidence_capture: true
    next_steps: ["analyze_results"]
    
  - id: "analyze_results"  
    name: "Analyze Hygiene Results"
    type: "decision"
    params:
      condition: "{{hygiene_score}} < 80"
    conditions:
      if_true: ["send_alert"]
      if_false: ["complete_workflow"]
    
  - id: "send_alert"
    name: "Send Hygiene Alert"
    type: "notify"
    params:
      channels: ["slack", "email"]
      message: "Pipeline hygiene score is {{hygiene_score}}% - Action required"
      recipients: ["sales_ops", "manager"]
    next_steps: ["complete_workflow"]
    
  - id: "complete_workflow"
    name: "Complete Workflow"
    type: "governance"
    params:
      action: "record"
      evidence_type: "pipeline_hygiene_audit"
      retention_days: 2555
"""

SAMPLE_FORECAST_APPROVAL_WORKFLOW = """
workflow_id: "forecast_approval_workflow"
name: "SaaS Forecast Approval Governance"  
description: "Automated forecast variance analysis with approval routing"
version: "1.0.0"

governance:
  policy_packs: ["forecast_governance", "sox_compliance"]
  evidence_required: true
  override_allowed: true

steps:
  - id: "calculate_variance"
    name: "Calculate Forecast Variance"
    type: "query"
    params:
      operator: "forecast_approval"
      config:
        variance_thresholds:
          low: 0.05
          medium: 0.15
          high: 0.25
        auto_approve_threshold: 0.05
    next_steps: ["check_variance_level"]
    
  - id: "check_variance_level"
    name: "Check Variance Level"
    type: "decision"
    params:
      condition: "{{variance_level}}"
    conditions:
      LOW: ["auto_approve"]
      MEDIUM: ["manager_approval"]
      HIGH: ["director_approval"]
      
  - id: "auto_approve"
    name: "Auto Approve Forecast"
    type: "action"
    params:
      action: "approve_forecast"
      reason: "Low variance - auto approved"
    next_steps: ["create_evidence"]
    
  - id: "manager_approval"
    name: "Route to Manager"
    type: "governance"
    params:
      approval_type: "manager"
      timeout_hours: 24
    next_steps: ["create_evidence"]
    
  - id: "director_approval"
    name: "Route to Director"
    type: "governance" 
    params:
      approval_type: "director"
      timeout_hours: 48
    next_steps: ["create_evidence"]
    
  - id: "create_evidence"
    name: "Create Evidence Pack"
    type: "governance"
    params:
      action: "create_evidence_pack"
      evidence_type: "forecast_approval_audit"
      retention_days: 2555
"""
