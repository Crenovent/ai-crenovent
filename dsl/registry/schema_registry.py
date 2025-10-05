"""
DSL Schema Registry Service
==========================

Task 7.1.7: Build schema registry integration
Provides schema validation and contract enforcement for DSL workflows
"""

import json
import yaml
import logging
import os
from typing import Dict, List, Any, Optional, Union
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime
import hashlib

# Import standard JSON schema validation
try:
    import jsonschema
    from jsonschema import Draft7Validator
    JSONSCHEMA_AVAILABLE = True
except ImportError:
    JSONSCHEMA_AVAILABLE = False

logger = logging.getLogger(__name__)

class SchemaType(Enum):
    """Schema types supported by the registry"""
    WORKFLOW = "workflow"
    STEP = "step"
    CONNECTOR = "connector"
    AGENT = "agent"
    POLICY = "policy"
    EVIDENCE = "evidence"

class SchemaStatus(Enum):
    """Schema lifecycle status"""
    DRAFT = "draft"
    APPROVED = "approved"
    DEPRECATED = "deprecated"
    RETIRED = "retired"

@dataclass
class SchemaVersion:
    """Schema version metadata"""
    version: str
    schema_content: Dict[str, Any]
    status: SchemaStatus
    created_at: datetime = field(default_factory=datetime.utcnow)
    created_by: str = "system"
    checksum: str = ""
    compatibility_level: str = "BACKWARD"  # BACKWARD, FORWARD, FULL, NONE
    
    def __post_init__(self):
        if not self.checksum:
            self.checksum = self._calculate_checksum()
    
    def _calculate_checksum(self) -> str:
        """Calculate SHA256 checksum of schema content"""
        schema_json = json.dumps(self.schema_content, sort_keys=True, separators=(',', ':'))
        return hashlib.sha256(schema_json.encode()).hexdigest()

@dataclass
class SchemaDefinition:
    """Complete schema definition with all versions"""
    schema_id: str
    schema_type: SchemaType
    name: str
    description: str
    versions: Dict[str, SchemaVersion] = field(default_factory=dict)
    latest_version: str = "1.0.0"
    tenant_id: int = 1300
    created_at: datetime = field(default_factory=datetime.utcnow)

class DSLSchemaRegistry:
    """
    DSL Schema Registry for validation and contract enforcement
    
    Implements Task 7.1.7: Build schema registry integration
    """
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.schemas: Dict[str, SchemaDefinition] = {}
        self._initialize_core_schemas()
    
    def _initialize_core_schemas(self):
        """Initialize core DSL schemas from JSON files"""
        
        # Load JSON schema files if available
        schema_dir = os.path.join(os.path.dirname(__file__), '..', 'schemas')
        
        # Try to load workflow schema from JSON file
        workflow_schema_path = os.path.join(schema_dir, 'workflow_schema.json')
        if os.path.exists(workflow_schema_path):
            try:
                with open(workflow_schema_path, 'r', encoding='utf-8') as f:
                    workflow_schema = json.load(f)
                    
                self.register_schema(
                    schema_id="core_workflow_v1",
                    schema_type=SchemaType.WORKFLOW,
                    name="Core Workflow Schema",
                    description="Core schema for DSL workflow definitions (from JSON file)",
                    schema_content=workflow_schema,
                    version="1.0.0"
                )
                
                self.logger.info("✅ Loaded workflow schema from JSON file")
                return  # Use JSON file schema instead of hardcoded
                
            except Exception as e:
                self.logger.warning(f"Failed to load JSON schema file: {e}, falling back to hardcoded schema")
        
        # Fallback to hardcoded schema if JSON file not available
        
        # Core workflow schema
        workflow_schema = {
            "$schema": "http://json-schema.org/draft-07/schema#",
            "type": "object",
            "title": "DSL Workflow Schema",
            "description": "Schema for DSL workflow definitions",
            "required": ["workflow_id", "name", "steps", "governance"],
            "properties": {
                "workflow_id": {
                    "type": "string",
                    "pattern": "^[a-zA-Z0-9_-]+$",
                    "description": "Unique workflow identifier"
                },
                "name": {
                    "type": "string",
                    "minLength": 1,
                    "maxLength": 255,
                    "description": "Human-readable workflow name"
                },
                "description": {
                    "type": "string",
                    "maxLength": 1000,
                    "description": "Workflow description"
                },
                "version": {
                    "type": "string",
                    "pattern": "^\\d+\\.\\d+\\.\\d+$",
                    "default": "1.0.0",
                    "description": "Semantic version"
                },
                "steps": {
                    "type": "array",
                    "minItems": 1,
                    "items": {"$ref": "#/definitions/step"},
                    "description": "Workflow steps"
                },
                "governance": {
                    "type": "object",
                    "required": ["tenant_id", "region_id", "sla_tier", "policy_pack_id"],
                    "properties": {
                        "tenant_id": {"type": "integer", "minimum": 1},
                        "region_id": {"type": "string", "enum": ["US", "EU", "APAC", "GLOBAL"]},
                        "sla_tier": {"type": "string", "enum": ["T0", "T1", "T2", "T3"]},
                        "policy_pack_id": {"type": "string"},
                        "trust_threshold": {"type": "number", "minimum": 0.0, "maximum": 1.0, "default": 0.8},
                        "trust_policy": {"type": "string", "enum": ["strict", "permissive", "adaptive"], "default": "strict"}
                    }
                },
                "metadata": {
                    "type": "object",
                    "description": "Additional workflow metadata"
                }
            },
            "definitions": {
                "step": {
                    "type": "object",
                    "required": ["id", "name", "type"],
                    "properties": {
                        "id": {
                            "type": "string",
                            "pattern": "^[a-zA-Z0-9_-]+$"
                        },
                        "name": {
                            "type": "string",
                            "minLength": 1,
                            "maxLength": 255
                        },
                        "type": {
                            "type": "string",
                            "enum": [
                                "query", "decision", "action", "notify", "governance",
                                "agent_call", "ml_decision", "subscription", "billing",
                                "customer_success", "churn_analysis", "revenue_recognition", "usage_metering"
                            ]
                        },
                        "params": {
                            "type": "object",
                            "description": "Step parameters"
                        },
                        "conditions": {
                            "type": "object",
                            "description": "Step conditions"
                        },
                        "next_steps": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "Next step IDs"
                        },
                        "governance": {
                            "type": "object",
                            "properties": {
                                "evidence_capture": {"type": "boolean", "default": true},
                                "consent_id": {"type": "string"},
                                "region_id": {"type": "string"},
                                "trust_threshold": {"type": "number", "minimum": 0.0, "maximum": 1.0},
                                "trust_metadata": {"type": "object"}
                            }
                        },
                        "retry_config": {
                            "type": "object",
                            "properties": {
                                "max_attempts": {"type": "integer", "minimum": 1, "maximum": 10, "default": 3},
                                "backoff_multiplier": {"type": "number", "minimum": 1.0, "default": 2.0},
                                "max_delay": {"type": "integer", "minimum": 1, "default": 300}
                            }
                        }
                    }
                }
            }
        }
        
        self.register_schema(
            schema_id="core_workflow_v1",
            schema_type=SchemaType.WORKFLOW,
            name="Core Workflow Schema",
            description="Core schema for DSL workflow definitions",
            schema_content=workflow_schema,
            version="1.0.0"
        )
        
        # Agent call step schema
        agent_schema = {
            "$schema": "http://json-schema.org/draft-07/schema#",
            "type": "object",
            "title": "Agent Call Step Schema",
            "description": "Schema for agent_call step type",
            "required": ["agent_id", "task", "governance"],
            "properties": {
                "agent_id": {
                    "type": "string",
                    "pattern": "^[a-zA-Z0-9_-]+$",
                    "description": "Agent identifier"
                },
                "task": {
                    "type": "string",
                    "minLength": 1,
                    "description": "Task description for the agent"
                },
                "context": {
                    "type": "object",
                    "description": "Context data for the agent"
                },
                "tools": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Available tools for the agent"
                },
                "governance": {
                    "type": "object",
                    "required": ["trust_threshold", "agent_data_consent"],
                    "properties": {
                        "trust_threshold": {"type": "number", "minimum": 0.0, "maximum": 1.0},
                        "agent_data_consent": {"type": "string"},
                        "agent_region": {"type": "string"},
                        "trust_metadata": {"type": "object"}
                    }
                }
            }
        }
        
        self.register_schema(
            schema_id="agent_call_step_v1",
            schema_type=SchemaType.STEP,
            name="Agent Call Step Schema",
            description="Schema for agent_call step parameters",
            schema_content=agent_schema,
            version="1.0.0"
        )
        
        # ML decision step schema
        ml_schema = {
            "$schema": "http://json-schema.org/draft-07/schema#",
            "type": "object",
            "title": "ML Decision Step Schema",
            "description": "Schema for ml_decision step type",
            "required": ["model_id", "input_features", "thresholds", "governance"],
            "properties": {
                "model_id": {
                    "type": "string",
                    "pattern": "^[a-zA-Z0-9_-]+$",
                    "description": "ML model identifier"
                },
                "input_features": {
                    "type": "object",
                    "description": "Input features for the model"
                },
                "thresholds": {
                    "type": "object",
                    "required": ["accept", "reject"],
                    "properties": {
                        "accept": {"type": "number", "minimum": 0.0, "maximum": 1.0},
                        "reject": {"type": "number", "minimum": 0.0, "maximum": 1.0}
                    }
                },
                "governance": {
                    "type": "object",
                    "required": ["model_consent", "trust_threshold"],
                    "properties": {
                        "model_consent": {"type": "string"},
                        "trust_threshold": {"type": "number", "minimum": 0.0, "maximum": 1.0},
                        "trust_metadata": {"type": "object"}
                    }
                }
            }
        }
        
        self.register_schema(
            schema_id="ml_decision_step_v1",
            schema_type=SchemaType.STEP,
            name="ML Decision Step Schema",
            description="Schema for ml_decision step parameters",
            schema_content=ml_schema,
            version="1.0.0"
        )
        
        self.logger.info("✅ Initialized core DSL schemas")
    
    def register_schema(
        self,
        schema_id: str,
        schema_type: SchemaType,
        name: str,
        description: str,
        schema_content: Dict[str, Any],
        version: str = "1.0.0",
        tenant_id: int = 1300,
        created_by: str = "system"
    ) -> bool:
        """Register a new schema or version"""
        
        try:
            # Create schema version
            schema_version = SchemaVersion(
                version=version,
                schema_content=schema_content,
                status=SchemaStatus.APPROVED,
                created_by=created_by
            )
            
            # Check if schema exists
            if schema_id in self.schemas:
                # Add new version
                schema_def = self.schemas[schema_id]
                schema_def.versions[version] = schema_version
                schema_def.latest_version = version
            else:
                # Create new schema
                schema_def = SchemaDefinition(
                    schema_id=schema_id,
                    schema_type=schema_type,
                    name=name,
                    description=description,
                    versions={version: schema_version},
                    latest_version=version,
                    tenant_id=tenant_id
                )
                self.schemas[schema_id] = schema_def
            
            self.logger.info(f"✅ Registered schema: {schema_id} v{version}")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Failed to register schema {schema_id}: {e}")
            return False
    
    def validate_workflow(self, workflow_data: Dict[str, Any], schema_id: str = "core_workflow_v1") -> Dict[str, Any]:
        """Validate workflow against schema using proper JSON Schema validation"""
        
        try:
            if schema_id not in self.schemas:
                return {
                    'valid': False,
                    'errors': [f"Schema {schema_id} not found"],
                    'warnings': []
                }
            
            schema_def = self.schemas[schema_id]
            latest_version = schema_def.versions[schema_def.latest_version]
            schema_content = latest_version.schema_content
            
            errors = []
            warnings = []
            
            # Use proper JSON Schema validation if available
            if JSONSCHEMA_AVAILABLE:
                try:
                    validator = Draft7Validator(schema_content)
                    validation_errors = list(validator.iter_errors(workflow_data))
                    
                    for error in validation_errors:
                        error_path = ".".join(str(p) for p in error.absolute_path) if error.absolute_path else "root"
                        errors.append(f"Schema error at {error_path}: {error.message}")
                    
                    self.logger.info(f"JSON Schema validation: {len(validation_errors)} errors found")
                    
                except Exception as e:
                    self.logger.warning(f"JSON Schema validation failed: {e}, falling back to basic validation")
                    # Fall back to basic validation
                    errors.extend(self._basic_validation(workflow_data, schema_content))
            else:
                # Fall back to basic validation if jsonschema not available
                errors.extend(self._basic_validation(workflow_data, schema_content))
            
            # Additional custom business logic validation
            custom_errors = self._custom_validation(workflow_data)
            errors.extend(custom_errors)
            
            return {
                'valid': len(errors) == 0,
                'errors': errors,
                'warnings': warnings,
                'schema_id': schema_id,
                'schema_version': schema_def.latest_version,
                'validation_method': 'jsonschema' if JSONSCHEMA_AVAILABLE else 'basic'
            }
            
        except Exception as e:
            self.logger.error(f"❌ Schema validation error: {e}")
            return {
                'valid': False,
                'errors': [f"Validation error: {str(e)}"],
                'warnings': []
            }
    
    def _basic_validation(self, workflow_data: Dict[str, Any], schema_content: Dict[str, Any]) -> List[str]:
        """Basic validation fallback when jsonschema is not available"""
        errors = []
        
        # Validate required fields
        required_fields = schema_content.get('required', [])
        for field in required_fields:
            if field not in workflow_data:
                errors.append(f"Missing required field: {field}")
        
        # Validate governance fields
        if 'governance' in workflow_data:
            governance = workflow_data['governance']
            gov_schema = schema_content.get('properties', {}).get('governance', {})
            gov_required = gov_schema.get('required', [])
            
            for field in gov_required:
                if field not in governance:
                    errors.append(f"Missing required governance field: {field}")
            
            # Validate trust threshold
            if 'trust_threshold' in governance:
                threshold = governance['trust_threshold']
                if not isinstance(threshold, (int, float)) or not (0.0 <= threshold <= 1.0):
                    errors.append(f"Invalid trust_threshold: {threshold} (must be number between 0.0-1.0)")
        
        # Validate steps
        if 'steps' in workflow_data:
            steps = workflow_data['steps']
            if not isinstance(steps, list) or len(steps) == 0:
                errors.append("Steps must be a non-empty array")
            else:
                for i, step in enumerate(steps):
                    step_errors = self._validate_step(step, i)
                    errors.extend(step_errors)
        
        return errors
    
    def _custom_validation(self, workflow_data: Dict[str, Any]) -> List[str]:
        """Custom business logic validation beyond JSON Schema"""
        errors = []
        
        # Validate step ID uniqueness
        if 'steps' in workflow_data:
            step_ids = []
            for step in workflow_data['steps']:
                if 'id' in step:
                    step_id = step['id']
                    if step_id in step_ids:
                        errors.append(f"Duplicate step ID: {step_id}")
                    step_ids.append(step_id)
        
        # Validate next_steps references
        if 'steps' in workflow_data:
            step_ids = {step.get('id') for step in workflow_data['steps'] if 'id' in step}
            for step in workflow_data['steps']:
                next_steps = step.get('next_steps', [])
                for next_step in next_steps:
                    if next_step not in step_ids:
                        errors.append(f"Step {step.get('id', 'unknown')} references non-existent step: {next_step}")
        
        # Validate industry-specific requirements
        industry = workflow_data.get('industry')
        compliance_frameworks = workflow_data.get('compliance_frameworks', [])
        
        if industry == 'SaaS' and 'SOX' not in compliance_frameworks:
            errors.append("SaaS workflows should include SOX compliance framework")
        
        if industry == 'Healthcare' and 'HIPAA' not in compliance_frameworks:
            errors.append("Healthcare workflows must include HIPAA compliance framework")
        
        return errors
    
    def _validate_step(self, step: Dict[str, Any], index: int) -> List[str]:
        """Validate individual step"""
        errors = []
        
        # Required step fields
        required_fields = ['id', 'name', 'type']
        for field in required_fields:
            if field not in step:
                errors.append(f"Step {index}: Missing required field '{field}'")
        
        # Validate step type
        if 'type' in step:
            step_type = step['type']
            valid_types = [
                'query', 'decision', 'action', 'notify', 'governance',
                'agent_call', 'ml_decision', 'subscription', 'billing',
                'customer_success', 'churn_analysis', 'revenue_recognition', 'usage_metering'
            ]
            if step_type not in valid_types:
                errors.append(f"Step {index}: Invalid step type '{step_type}'")
            
            # Type-specific validation
            if step_type == 'agent_call':
                errors.extend(self._validate_agent_step(step, index))
            elif step_type == 'ml_decision':
                errors.extend(self._validate_ml_step(step, index))
        
        return errors
    
    def _validate_agent_step(self, step: Dict[str, Any], index: int) -> List[str]:
        """Validate agent_call step"""
        errors = []
        params = step.get('params', {})
        
        if 'agent_id' not in params:
            errors.append(f"Step {index}: agent_call requires agent_id parameter")
        
        governance = step.get('governance', {})
        if 'trust_threshold' not in governance:
            errors.append(f"Step {index}: agent_call requires trust_threshold in governance")
        
        return errors
    
    def _validate_ml_step(self, step: Dict[str, Any], index: int) -> List[str]:
        """Validate ml_decision step"""
        errors = []
        params = step.get('params', {})
        
        if 'model_id' not in params:
            errors.append(f"Step {index}: ml_decision requires model_id parameter")
        
        if 'thresholds' not in params:
            errors.append(f"Step {index}: ml_decision requires thresholds parameter")
        else:
            thresholds = params['thresholds']
            if 'accept' not in thresholds or 'reject' not in thresholds:
                errors.append(f"Step {index}: ml_decision thresholds must include 'accept' and 'reject'")
        
        return errors
    
    def get_schema(self, schema_id: str, version: Optional[str] = None) -> Optional[Dict[str, Any]]:
        """Get schema by ID and version"""
        
        if schema_id not in self.schemas:
            return None
        
        schema_def = self.schemas[schema_id]
        target_version = version or schema_def.latest_version
        
        if target_version not in schema_def.versions:
            return None
        
        return schema_def.versions[target_version].schema_content
    
    def list_schemas(self, schema_type: Optional[SchemaType] = None, tenant_id: Optional[int] = None) -> List[Dict[str, Any]]:
        """List all schemas with optional filtering"""
        
        schemas = []
        for schema_id, schema_def in self.schemas.items():
            if schema_type and schema_def.schema_type != schema_type:
                continue
            if tenant_id and schema_def.tenant_id != tenant_id:
                continue
            
            schemas.append({
                'schema_id': schema_id,
                'name': schema_def.name,
                'description': schema_def.description,
                'type': schema_def.schema_type.value,
                'latest_version': schema_def.latest_version,
                'versions': list(schema_def.versions.keys()),
                'tenant_id': schema_def.tenant_id,
                'created_at': schema_def.created_at.isoformat()
            })
        
        return schemas
    
    def check_compatibility(self, schema_id: str, new_schema: Dict[str, Any]) -> Dict[str, Any]:
        """Check schema compatibility for evolution"""
        
        if schema_id not in self.schemas:
            return {'compatible': True, 'issues': []}
        
        schema_def = self.schemas[schema_id]
        current_schema = schema_def.versions[schema_def.latest_version].schema_content
        
        issues = []
        
        # Check for breaking changes
        current_required = set(current_schema.get('required', []))
        new_required = set(new_schema.get('required', []))
        
        # New required fields are breaking changes
        new_required_fields = new_required - current_required
        if new_required_fields:
            issues.append(f"Breaking change: New required fields {list(new_required_fields)}")
        
        # Removed required fields are breaking changes
        removed_required_fields = current_required - new_required
        if removed_required_fields:
            issues.append(f"Breaking change: Removed required fields {list(removed_required_fields)}")
        
        return {
            'compatible': len(issues) == 0,
            'issues': issues
        }

# Global schema registry instance
schema_registry = DSLSchemaRegistry()

def get_schema_registry() -> DSLSchemaRegistry:
    """Get the global schema registry instance"""
    return schema_registry
