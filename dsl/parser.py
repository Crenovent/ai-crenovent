"""
Enhanced DSL Parser - Converts YAML workflows into executable workflow objects
Enhanced with governance-by-design validation and orchestration DSL grammar
Tasks 6.4-T01 to T08: Orchestration DSL grammar, primitives, governance guards, evidence hooks
"""

import yaml
import json
import jsonschema
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field
from datetime import datetime
import logging
import asyncio

from .governance.governance_metadata_validator import (
    get_governance_validator, 
    ValidationContext, 
    EnforcementMode,
    ValidationResult
)

logger = logging.getLogger(__name__)

@dataclass
class DSLStep:
    """Represents a single step in a DSL workflow"""
    id: str
    type: str
    params: Dict[str, Any]
    outputs: Dict[str, str] = field(default_factory=dict)
    governance: Dict[str, Any] = field(default_factory=dict)
    next_steps: List[str] = field(default_factory=list)
    
    def __post_init__(self):
        # Ensure governance is present
        if not self.governance:
            self.governance = {
                'policy_id': 'default_policy',
                'evidence_capture': True
            }

@dataclass 
class DSLWorkflow:
    """Represents a complete DSL workflow with governance metadata"""
    workflow_id: str
    name: str
    module: str
    automation_type: str  # RBA, RBIA, AALA
    version: str
    steps: List[DSLStep]
    governance: Dict[str, Any]
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    # Governance metadata (Task 7.1.3)
    tenant_id: Optional[int] = None
    region_id: Optional[str] = None
    sla_tier: Optional[str] = None
    policy_pack_id: Optional[str] = None
    trust_score_threshold: Optional[float] = None
    consent_id: Optional[str] = None
    evidence_pack_required: bool = True
    governance_validation_passed: bool = False
    governance_validation_errors: List[str] = field(default_factory=list)
    
    def validate(self) -> List[str]:
        """Validate workflow structure and return errors"""
        errors = []
        
        # Check required fields
        if not self.workflow_id:
            errors.append("workflow_id is required")
        if not self.steps:
            errors.append("At least one step is required")
            
        # Validate automation type
        if self.automation_type not in ['RBA', 'RBIA', 'AALA']:
            errors.append(f"Invalid automation_type: {self.automation_type}")
            
        # Validate steps
        step_ids = set()
        for step in self.steps:
            if step.id in step_ids:
                errors.append(f"Duplicate step ID: {step.id}")
            step_ids.add(step.id)
            
            # Check step type is supported
            if step.type not in ['query', 'decision', 'ml_decision', 'agent_call', 'notify', 'governance', 'policy_query']:
                errors.append(f"Unsupported step type: {step.type}")
                
        return errors

class DSLParser:
    """
    Parses YAML DSL files into executable workflow objects
    Enhanced with governance-by-design validation (Task 7.1.3)
    """
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self._load_schemas()
        self.governance_validator = get_governance_validator()
    
    def _load_schemas(self):
        """Load JSON schemas for validation"""
        # DSL Schema for validation
        self.workflow_schema = {
            "type": "object",
            "required": ["workflow_id", "module", "automation_type", "steps"],
            "properties": {
                "workflow_id": {"type": "string"},
                "name": {"type": "string"},
                "module": {"type": "string"},
                "automation_type": {"type": "string", "enum": ["RBA", "RBIA", "AALA"]},
                "version": {"type": "string"},
                "steps": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "required": ["id", "type", "params"],
                        "properties": {
                            "id": {"type": "string"},
                            "type": {"type": "string", "enum": ["query", "decision", "ml_decision", "agent_call", "notify", "governance", "policy_query"]},
                            "params": {"type": "object"},
                            "outputs": {"type": "object"},
                            "governance": {"type": "object"}
                        }
                    }
                },
                "governance": {"type": "object"}
            }
        }
    
    def parse_yaml(self, yaml_content: str) -> DSLWorkflow:
        """
        Parse YAML content into DSLWorkflow object
        
        Args:
            yaml_content: Raw YAML string
            
        Returns:
            DSLWorkflow object
            
        Raises:
            ValueError: If parsing fails or validation errors
        """
        try:
            # Parse YAML
            data = yaml.safe_load(yaml_content)
            
            # Validate against schema
            jsonschema.validate(data, self.workflow_schema)
            
            # Convert to DSLWorkflow
            workflow = self._convert_to_workflow(data)
            
            # Validate business logic
            errors = workflow.validate()
            if errors:
                raise ValueError(f"Workflow validation failed: {', '.join(errors)}")
                
            self.logger.info(f"Successfully parsed workflow: {workflow.workflow_id}")
            return workflow
            
        except yaml.YAMLError as e:
            raise ValueError(f"YAML parsing error: {e}")
        except jsonschema.ValidationError as e:
            raise ValueError(f"Schema validation error: {e.message}")
        except Exception as e:
            self.logger.error(f"Unexpected parsing error: {e}")
            raise ValueError(f"Parsing failed: {e}")
    
    def _convert_to_workflow(self, data: Dict[str, Any]) -> DSLWorkflow:
        """Convert parsed YAML data to DSLWorkflow object"""
        
        # Parse steps
        steps = []
        for step_data in data.get('steps', []):
            step = DSLStep(
                id=step_data['id'],
                type=step_data['type'],
                params=step_data['params'],
                outputs=step_data.get('outputs', {}),
                governance=step_data.get('governance', {}),
                next_steps=step_data.get('next_steps', [])
            )
            steps.append(step)
        
        # Create workflow
        workflow = DSLWorkflow(
            workflow_id=data['workflow_id'],
            name=data.get('name', data['workflow_id']),
            module=data['module'],
            automation_type=data['automation_type'],
            version=data.get('version', '1.0.0'),
            steps=steps,
            governance=data.get('governance', {}),
            metadata=data.get('metadata', {})
        )
        
        return workflow
    
    def parse_file(self, file_path: str) -> DSLWorkflow:
        """Parse DSL workflow from file"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            return self.parse_yaml(content)
        except FileNotFoundError:
            raise ValueError(f"Workflow file not found: {file_path}")
        except Exception as e:
            raise ValueError(f"Failed to read workflow file: {e}")

    def workflow_to_yaml(self, workflow: DSLWorkflow) -> str:
        """Convert DSLWorkflow back to YAML string"""
        data = {
            'workflow_id': workflow.workflow_id,
            'name': workflow.name,
            'module': workflow.module,
            'automation_type': workflow.automation_type,
            'version': workflow.version,
            'steps': [
                {
                    'id': step.id,
                    'type': step.type,
                    'params': step.params,
                    'outputs': step.outputs,
                    'governance': step.governance
                }
                for step in workflow.steps
            ],
            'governance': workflow.governance,
            'metadata': workflow.metadata
        }
        
        return yaml.dump(data, default_flow_style=False, sort_keys=False)
    
    async def parse_yaml_with_governance(
        self, 
        yaml_content: str, 
        tenant_id: Optional[int] = None,
        user_id: Optional[int] = None,
        enforcement_mode: EnforcementMode = EnforcementMode.COMPLIANCE
    ) -> DSLWorkflow:
        """
        Parse YAML content with governance validation (Task 7.1.3)
        
        Args:
            yaml_content: Raw YAML string
            tenant_id: Tenant identifier for multi-tenant isolation
            user_id: User identifier for audit trail
            enforcement_mode: Governance enforcement mode
            
        Returns:
            DSLWorkflow object with governance validation
            
        Raises:
            ValueError: If parsing or governance validation fails
        """
        try:
            # Parse YAML
            data = yaml.safe_load(yaml_content)
            
            # Validate against schema
            jsonschema.validate(data, self.workflow_schema)
            
            # Convert to DSLWorkflow
            workflow = self._convert_to_workflow(data)
            
            # Validate business logic
            errors = workflow.validate()
            if errors:
                raise ValueError(f"Workflow validation failed: {', '.join(errors)}")
            
            # Extract governance metadata
            governance_metadata = self._extract_governance_metadata(data, tenant_id, user_id)
            
            # Create validation context
            context = ValidationContext(
                tenant_id=tenant_id,
                user_id=user_id,
                workflow_type=workflow.automation_type,
                industry_overlay=governance_metadata.get('industry_overlay'),
                enforcement_mode=enforcement_mode
            )
            
            # Perform governance validation
            is_valid, validation_errors = await self.governance_validator.validate_governance_metadata(
                governance_metadata, context
            )
            
            # Update workflow with governance validation results
            workflow.governance_validation_passed = is_valid
            workflow.governance_validation_errors = [str(error.message) for error in validation_errors]
            
            # Apply governance metadata to workflow
            self._apply_governance_metadata(workflow, governance_metadata)
            
            # Log validation results
            if is_valid:
                self.logger.info(f"✅ Governance validation passed for workflow: {workflow.workflow_id}")
            else:
                error_summary = f"❌ Governance validation failed for workflow: {workflow.workflow_id} - {len(validation_errors)} errors"
                self.logger.warning(error_summary)
                
                # Determine if we should block execution
                blocking_errors = [e for e in validation_errors if e.enforcement_action == "fail_closed"]
                if blocking_errors and enforcement_mode != EnforcementMode.PERMISSIVE:
                    raise ValueError(f"Governance validation failed: {'; '.join([e.message for e in blocking_errors])}")
            
            return workflow
            
        except yaml.YAMLError as e:
            raise ValueError(f"YAML parsing error: {e}")
        except jsonschema.ValidationError as e:
            raise ValueError(f"Schema validation error: {e.message}")
        except Exception as e:
            self.logger.error(f"Unexpected parsing error: {e}")
            raise ValueError(f"Parsing failed: {e}")
    
    def _extract_governance_metadata(
        self, 
        data: Dict[str, Any], 
        tenant_id: Optional[int] = None,
        user_id: Optional[int] = None
    ) -> Dict[str, Any]:
        """Extract governance metadata from workflow data"""
        governance_metadata = {}
        
        # Extract from top-level metadata
        metadata = data.get('metadata', {})
        governance_metadata.update(metadata.get('governance', {}))
        
        # Extract from governance section
        governance = data.get('governance', {})
        governance_metadata.update(governance)
        
        # Add context information
        if tenant_id:
            governance_metadata['tenant_id'] = tenant_id
        if user_id:
            governance_metadata['created_by_user_id'] = user_id
            
        # Extract workflow-level information
        governance_metadata['workflow_type'] = data.get('automation_type')
        governance_metadata['workflow_id'] = data.get('workflow_id')
        governance_metadata['module'] = data.get('module')
        
        # Set defaults if not provided
        if 'sla_tier' not in governance_metadata:
            governance_metadata['sla_tier'] = 'T1'  # Default to T1
        if 'trust_score_threshold' not in governance_metadata:
            governance_metadata['trust_score_threshold'] = 0.7  # Default trust threshold
        if 'evidence_pack_required' not in governance_metadata:
            governance_metadata['evidence_pack_required'] = True  # Default to required
            
        return governance_metadata
    
    def _apply_governance_metadata(self, workflow: DSLWorkflow, metadata: Dict[str, Any]):
        """Apply governance metadata to workflow object"""
        workflow.tenant_id = metadata.get('tenant_id')
        workflow.region_id = metadata.get('region_id')
        workflow.sla_tier = metadata.get('sla_tier')
        workflow.policy_pack_id = metadata.get('policy_pack_id')
        workflow.trust_score_threshold = metadata.get('trust_score_threshold')
        workflow.consent_id = metadata.get('consent_id')
        workflow.evidence_pack_required = metadata.get('evidence_pack_required', True)
