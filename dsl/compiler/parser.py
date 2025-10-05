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

# Import schema registry for validation (Task 7.1.7)
try:
    from ..registry.schema_registry import get_schema_registry
except ImportError:
    # Fallback if schema registry is not available
    def get_schema_registry():
        return None

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
        
        # Schema registry integration (Task 7.1.7)
        self.schema_registry = get_schema_registry()
        self.schema_validation_enabled = self.schema_registry is not None
        
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
                
            # Schema validation (Task 7.1.7)
            if self.schema_validation_enabled:
                schema_results = self.schema_registry.validate_workflow(workflow_dict)
                if not schema_results['valid']:
                    raise CompilationError(f"Schema validation failed: {schema_results['errors']}")
                self.logger.info(f"✅ Schema validation passed: {schema_results['schema_id']} v{schema_results['schema_version']}")
            
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
            
            # Consent validation (Task 7.1.8)
            consent_results = self._validate_consent_compliance(ast)
            if not consent_results['compliant']:
                raise CompilationError(f"Consent validation failed: {consent_results['violations']}")
            
            # Residency enforcement (Task 7.1.9)
            residency_results = self._validate_residency_compliance(ast)
            if not residency_results['compliant']:
                raise CompilationError(f"Residency validation failed: {residency_results['violations']}")
            
            # Trust score annotations (Task 7.1.10)
            trust_results = self._validate_and_annotate_trust_scores(ast)
            if not trust_results['compliant']:
                raise CompilationError(f"Trust score validation failed: {trust_results['violations']}")
            
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
    
    def _validate_consent_compliance(self, ast: DSLWorkflowAST) -> Dict[str, Any]:
        """
        Validate consent compliance in DSL workflow (Task 7.1.8)
        
        Args:
            ast: DSL workflow AST
            
        Returns:
            Dict with consent validation results
        """
        violations = []
        
        try:
            # Get region and industry from governance metadata
            governance = ast.governance or {}
            region_id = governance.get('region_id', 'GLOBAL')
            industry_overlay = governance.get('industry_overlay', 'GENERIC')
            
            # Define regions requiring consent validation
            consent_required_regions = ['EU', 'UK', 'IN', 'CA', 'BR']  # GDPR, UK-GDPR, DPDP, PIPEDA, LGPD
            
            # Check if consent is required for this region
            consent_required = region_id in consent_required_regions
            
            if consent_required:
                # Validate consent_id is present
                consent_id = governance.get('consent_id')
                if not consent_id:
                    violations.append(f"Consent ID required for region {region_id} but not provided")
                
                # Validate consent scope covers workflow operations
                consent_scope = governance.get('consent_scope', [])
                workflow_operations = self._extract_workflow_operations(ast)
                
                for operation in workflow_operations:
                    if operation not in consent_scope:
                        violations.append(f"Operation '{operation}' not covered by consent scope")
                
                # Validate consent expiry
                consent_expiry = governance.get('consent_expiry')
                if consent_expiry:
                    try:
                        from datetime import datetime
                        expiry_date = datetime.fromisoformat(consent_expiry.replace('Z', '+00:00'))
                        if expiry_date < datetime.utcnow().replace(tzinfo=expiry_date.tzinfo):
                            violations.append(f"Consent expired on {consent_expiry}")
                    except ValueError:
                        violations.append(f"Invalid consent expiry format: {consent_expiry}")
                
                # Industry-specific consent requirements
                if industry_overlay == 'HEALTHCARE':
                    # HIPAA requires explicit consent for PHI processing
                    phi_processing = any(
                        step.params and step.params.get('data_type') == 'phi' 
                        for step in ast.steps
                    )
                    if phi_processing and 'phi_processing' not in consent_scope:
                        violations.append("PHI processing requires explicit consent for healthcare workflows")
                
                elif industry_overlay == 'FINANCIAL':
                    # Financial data requires specific consent categories
                    financial_operations = ['credit_check', 'payment_processing', 'account_analysis']
                    for op in financial_operations:
                        if op in workflow_operations and op not in consent_scope:
                            violations.append(f"Financial operation '{op}' requires explicit consent")
            
            self.logger.info(f"✅ Consent validation completed: {len(violations)} violations found")
            
        except Exception as e:
            self.logger.error(f"❌ Consent validation failed: {e}")
            violations.append(f"Consent validation error: {str(e)}")
        
        return {
            'compliant': len(violations) == 0,
            'violations': violations
        }
    
    def _extract_workflow_operations(self, ast: DSLWorkflowAST) -> List[str]:
        """Extract operations performed by workflow for consent validation"""
        operations = set()
        
        for step in ast.steps:
            step_type = step.type.value if hasattr(step.type, 'value') else str(step.type)
            step_params = step.params or {}
            
            # Map step types to operations
            if step_type == 'query':
                source = step_params.get('source', 'unknown')
                operations.add(f"data_query_{source}")
                
                # Specific data types
                if 'customer' in str(step_params).lower():
                    operations.add('customer_data_access')
                if 'billing' in str(step_params).lower():
                    operations.add('billing_data_access')
                if 'usage' in str(step_params).lower():
                    operations.add('usage_data_access')
            
            elif step_type == 'decision':
                operations.add('automated_decision_making')
                
                # Check for profiling
                if any(keyword in str(step_params).lower() for keyword in ['score', 'risk', 'profile', 'segment']):
                    operations.add('profiling')
            
            elif step_type == 'ml_decision':
                operations.add('automated_decision_making')
                operations.add('profiling')
                operations.add('ml_processing')
            
            elif step_type == 'notify':
                operations.add('communication')
                
                # Check for marketing communications
                if 'marketing' in str(step_params).lower():
                    operations.add('marketing_communication')
            
            elif step_type == 'agent_call':
                operations.add('ai_processing')
                operations.add('automated_decision_making')
            
            # Check for specific data types in parameters
            data_type = step_params.get('data_type')
            if data_type == 'phi':
                operations.add('phi_processing')
            elif data_type == 'financial':
                operations.add('financial_data_processing')
            elif data_type == 'biometric':
                operations.add('biometric_processing')
        
        return list(operations)
    
    def _validate_residency_compliance(self, ast: DSLWorkflowAST) -> Dict[str, Any]:
        """
        Validate residency compliance in DSL workflow (Task 7.1.9)
        
        Args:
            ast: DSL workflow AST
            
        Returns:
            Dict with residency validation results
        """
        violations = []
        
        try:
            governance = ast.governance or {}
            region_id = governance.get('region_id', 'GLOBAL')
            tenant_id = governance.get('tenant_id')
            
            # Validate region-specific requirements
            if region_id == 'EU':
                # GDPR requirements
                if not governance.get('gdpr_compliant', False):
                    violations.append("EU region requires GDPR compliance flag")
                
                # Data residency check
                data_residency = governance.get('data_residency', 'GLOBAL')
                if data_residency not in ['EU', 'GLOBAL']:
                    violations.append(f"EU region requires EU or GLOBAL data residency, got {data_residency}")
            
            elif region_id == 'IN':
                # DPDP requirements
                if not governance.get('dpdp_compliant', False):
                    violations.append("IN region requires DPDP compliance flag")
                
                # Data localization check for critical personal data
                has_critical_data = any(
                    step.params and step.params.get('data_sensitivity') == 'critical'
                    for step in ast.steps
                )
                if has_critical_data:
                    data_residency = governance.get('data_residency', 'GLOBAL')
                    if data_residency != 'IN':
                        violations.append("Critical personal data in IN region must have IN data residency")
            
            elif region_id == 'US':
                # State-specific requirements (simplified)
                state = governance.get('state')
                if state == 'CA' and not governance.get('ccpa_compliant', False):
                    violations.append("California requires CCPA compliance flag")
            
            # Cross-border data transfer validation
            for step in ast.steps:
                step_params = step.params or {}
                target_region = step_params.get('target_region')
                
                if target_region and target_region != region_id:
                    # Check if cross-border transfer is allowed
                    transfer_mechanism = governance.get('cross_border_transfer_mechanism')
                    if not transfer_mechanism:
                        violations.append(f"Cross-border data transfer to {target_region} requires transfer mechanism")
                    
                    # Specific transfer validations
                    if region_id == 'EU' and target_region not in ['EU', 'UK']:
                        adequacy_decision = governance.get('adequacy_decision', False)
                        standard_contractual_clauses = governance.get('standard_contractual_clauses', False)
                        
                        if not (adequacy_decision or standard_contractual_clauses):
                            violations.append(f"EU to {target_region} transfer requires adequacy decision or SCCs")
            
            self.logger.info(f"✅ Residency validation completed: {len(violations)} violations found")
            
        except Exception as e:
            self.logger.error(f"❌ Residency validation failed: {e}")
            violations.append(f"Residency validation error: {str(e)}")
        
        return {
            'compliant': len(violations) == 0,
            'violations': violations
        }
    
    def _validate_and_annotate_trust_scores(self, ast: DSLWorkflowAST) -> Dict[str, Any]:
        """
        Validate and annotate trust scores in DSL (Task 7.1.10)
        
        Adds reliability metadata for agent orchestration and integrates with trust scoring API
        """
        violations = []
        
        try:
            # Check workflow-level trust configuration
            governance = ast.governance or {}
            trust_threshold = governance.get('trust_threshold', 0.8)
            trust_policy = governance.get('trust_policy', 'strict')
            
            # Validate trust threshold range
            if not (0.0 <= trust_threshold <= 1.0):
                violations.append(f"Invalid trust_threshold {trust_threshold}: must be between 0.0 and 1.0")
            
            # Check each step for trust score requirements
            for step in ast.steps:
                step_params = step.params or {}
                step_type = step.type.value if hasattr(step.type, 'value') else str(step.type)
                governance_block = step.governance or {}
                
                # Agent calls and ML decisions require trust scores
                if step_type in ['agent_call', 'ml_decision']:
                    
                    # Check for trust_threshold at step level
                    step_trust_threshold = governance_block.get('trust_threshold', trust_threshold)
                    if not (0.0 <= step_trust_threshold <= 1.0):
                        violations.append(f"Step {step.step_id}: Invalid trust_threshold {step_trust_threshold}")
                    
                    # Agent calls require agent trust metadata
                    if step_type == 'agent_call':
                        agent_id = step_params.get('agent_id', '')
                        if not agent_id:
                            violations.append(f"Step {step.step_id}: Missing agent_id for trust scoring")
                        
                        # Annotate with trust requirements
                        if 'trust_metadata' not in governance_block:
                            governance_block['trust_metadata'] = {
                                'required_trust_score': step_trust_threshold,
                                'trust_policy': trust_policy,
                                'fallback_on_low_trust': True,
                                'trust_validation_required': True,
                                'agent_id': agent_id,
                                'trust_calculation_method': 'weighted_average',
                                'trust_factors': ['execution_history', 'error_rate', 'sla_compliance', 'override_frequency']
                            }
                    
                    # ML decisions require model trust metadata
                    if step_type == 'ml_decision':
                        model_id = step_params.get('model_id', '')
                        if not model_id:
                            violations.append(f"Step {step.step_id}: Missing model_id for trust scoring")
                        
                        # Annotate with model trust requirements
                        if 'trust_metadata' not in governance_block:
                            governance_block['trust_metadata'] = {
                                'required_model_trust_score': step_trust_threshold,
                                'model_confidence_threshold': step_params.get('thresholds', {}).get('accept', 0.8),
                                'trust_policy': trust_policy,
                                'fallback_on_low_trust': True,
                                'model_validation_required': True,
                                'model_id': model_id,
                                'trust_calculation_method': 'confidence_weighted',
                                'trust_factors': ['model_accuracy', 'prediction_confidence', 'data_quality', 'drift_detection']
                            }
                
                # Query steps may require data source trust validation
                elif step_type == 'query':
                    source = step_params.get('source', '')
                    if source in ['api', 'external']:
                        # External data sources require trust validation
                        if 'trust_metadata' not in governance_block:
                            governance_block['trust_metadata'] = {
                                'data_source_trust_required': True,
                                'source_validation_policy': trust_policy,
                                'trust_threshold': trust_threshold,
                                'source': source,
                                'trust_factors': ['data_freshness', 'source_reliability', 'schema_compliance']
                            }
                
                # All steps get basic trust tracking
                if 'trust_metadata' not in governance_block:
                    governance_block['trust_metadata'] = {
                        'trust_tracking_enabled': True,
                        'execution_trust_logging': True,
                        'step_type': step_type,
                        'trust_factors': ['execution_success', 'latency', 'resource_usage']
                    }
                
                # Update the step with annotated governance
                step.governance = governance_block
            
            # Annotate workflow-level trust metadata
            if 'trust_metadata' not in governance:
                governance['trust_metadata'] = {
                    'workflow_trust_threshold': trust_threshold,
                    'trust_policy': trust_policy,
                    'trust_aggregation_method': 'weighted_average',
                    'trust_validation_enabled': True,
                    'trust_score_required_for_execution': True,
                    'trust_calculation_factors': {
                        'step_success_rate': 0.4,
                        'sla_compliance': 0.3,
                        'override_frequency': 0.2,
                        'execution_latency': 0.1
                    },
                    'trust_score_cache_ttl': 300,  # 5 minutes
                    'trust_score_api_integration': True
                }
                ast.governance = governance
            
            self.logger.info(f"Trust score validation completed: {len(violations)} violations found")
            
        except Exception as e:
            self.logger.error(f"Trust score validation error: {e}")
            violations.append(f"Trust score validation failed: {str(e)}")
        
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
