"""
Override Justification Schema Validator - Task 6.2.22
======================================================

Override allowed & justification schema present
- Linter/enforcer that detects nodes with override_allowed: true
- Verifies presence of override_justification_schema with required fields (who, why, evidence pointers)
- Ensures overrides will be recorded in override ledger format (hash-chained record)
- Provides JSON Schema template generator for common override types

Dependencies: Task 6.2.4 (IR), override_service.py, override_ledger.py
Outputs: Override justification validation → ensures audit-ready override tracking
"""

from typing import Dict, List, Any, Optional, Set, Tuple
from dataclasses import dataclass, field
from enum import Enum
import logging
import json
from datetime import datetime

logger = logging.getLogger(__name__)

@dataclass
class OverrideJustificationError:
    """Override justification validation error"""
    error_code: str
    node_id: str
    model_id: Optional[str]
    severity: str  # "error", "warning", "info"
    message: str
    expected: str
    actual: str
    autofix_available: bool = False
    suggested_schema: Optional[Dict[str, Any]] = None
    source_location: Optional[Dict[str, Any]] = None

@dataclass
class OverrideSchemaTemplate:
    """Template for override justification schema"""
    template_name: str
    template_type: str  # "financial", "compliance", "risk_assessment", "manual_review"
    description: str
    required_fields: List[str]
    optional_fields: List[str]
    schema: Dict[str, Any]
    validation_rules: List[str]

class OverrideValidationSeverity(Enum):
    """Severity levels for override validation"""
    ERROR = "error"
    WARNING = "warning"
    INFO = "info"

class OverrideType(Enum):
    """Types of overrides that require justification"""
    FINANCIAL_DECISION = "financial_decision"
    RISK_ASSESSMENT = "risk_assessment"
    COMPLIANCE_OVERRIDE = "compliance_override"
    MANUAL_REVIEW = "manual_review"
    EMERGENCY_OVERRIDE = "emergency_override"
    POLICY_EXCEPTION = "policy_exception"

# Task 6.2.22: Override Justification Schema Validator
class OverrideJustificationValidator:
    """Validates override justification schemas for audit readiness"""
    
    def __init__(self):
        self.validation_errors: List[OverrideJustificationError] = []
        self.schema_templates: Dict[str, OverrideSchemaTemplate] = {}
        
        # Initialize schema templates
        self._init_schema_templates()
    
    def validate_override_justification_schemas(self, ir_graph) -> List[OverrideJustificationError]:
        """Validate override justification schemas for all nodes with override_allowed"""
        from .ir import IRGraph, IRNodeType
        
        self.validation_errors.clear()
        
        # Check all nodes for override configuration
        for node in ir_graph.nodes:
            if self._has_override_allowed(node):
                self._validate_node_override_schema(node)
        
        return self.validation_errors
    
    def _has_override_allowed(self, node) -> bool:
        """Check if node has override_allowed: true"""
        # Check direct parameter
        if node.parameters.get('override_allowed', False):
            return True
        
        # Check in governance configuration
        governance = node.parameters.get('governance', {})
        if governance.get('override_allowed', False):
            return True
        
        # Check in policies
        for policy in node.policies:
            if policy.policy_type == 'override' and policy.parameters.get('allowed', False):
                return True
        
        # Check metadata
        if node.metadata.get('override_allowed', False):
            return True
        
        return False
    
    def _validate_node_override_schema(self, node):
        """Validate override justification schema for a single node"""
        model_id = self._extract_model_id(node)
        
        # Check for justification schema
        justification_schema = self._extract_justification_schema(node)
        
        if not justification_schema:
            self._report_missing_schema(node, model_id)
            return
        
        # Validate schema completeness
        self._validate_schema_completeness(node, model_id, justification_schema)
        
        # Validate schema structure
        self._validate_schema_structure(node, model_id, justification_schema)
        
        # Validate ledger integration
        self._validate_ledger_integration(node, model_id, justification_schema)
    
    def _extract_justification_schema(self, node) -> Optional[Dict[str, Any]]:
        """Extract override justification schema from node"""
        # Check direct schema parameter
        if 'override_justification_schema' in node.parameters:
            return node.parameters['override_justification_schema']
        
        # Check in governance configuration
        governance = node.parameters.get('governance', {})
        if 'justification_schema' in governance:
            return governance['justification_schema']
        
        # Check in override configuration
        override_config = node.parameters.get('override_config', {})
        if 'justification_schema' in override_config:
            return override_config['justification_schema']
        
        # Check metadata
        if 'justification_schema' in node.metadata:
            return node.metadata['justification_schema']
        
        return None
    
    def _report_missing_schema(self, node, model_id: Optional[str]):
        """Report missing justification schema"""
        override_type = self._infer_override_type(node, model_id)
        suggested_template = self._get_template_for_override_type(override_type)
        
        self.validation_errors.append(OverrideJustificationError(
            error_code="OJ001",
            node_id=node.id,
            model_id=model_id,
            severity=OverrideValidationSeverity.ERROR.value,
            message="Node with override_allowed missing justification schema",
            expected="override_justification_schema with required fields (who, why, evidence)",
            actual="No justification schema found",
            autofix_available=True,
            suggested_schema=suggested_template.schema if suggested_template else None
        ))
    
    def _validate_schema_completeness(self, node, model_id: Optional[str], schema: Dict[str, Any]):
        """Validate completeness of justification schema"""
        required_core_fields = ['who', 'why', 'evidence', 'timestamp', 'override_type']
        
        # Check for required core fields
        schema_fields = schema.get('properties', {}).keys() if 'properties' in schema else schema.keys()
        
        for required_field in required_core_fields:
            if required_field not in schema_fields:
                self.validation_errors.append(OverrideJustificationError(
                    error_code="OJ002",
                    node_id=node.id,
                    model_id=model_id,
                    severity=OverrideValidationSeverity.ERROR.value,
                    message=f"Justification schema missing required field: {required_field}",
                    expected=f"Required field '{required_field}' in schema",
                    actual="Field not found in schema",
                    autofix_available=True,
                    suggested_schema=self._generate_field_addition(schema, required_field)
                ))
        
        # Check for audit trail fields
        audit_fields = ['approver_id', 'approval_timestamp', 'review_status']
        missing_audit_fields = [field for field in audit_fields if field not in schema_fields]
        
        if missing_audit_fields:
            self.validation_errors.append(OverrideJustificationError(
                error_code="OJ003",
                node_id=node.id,
                model_id=model_id,
                severity=OverrideValidationSeverity.WARNING.value,
                message=f"Justification schema missing audit fields: {missing_audit_fields}",
                expected=f"Audit fields for compliance: {audit_fields}",
                actual=f"Missing: {missing_audit_fields}",
                autofix_available=True,
                suggested_schema=self._generate_audit_fields_addition(schema, missing_audit_fields)
            ))
    
    def _validate_schema_structure(self, node, model_id: Optional[str], schema: Dict[str, Any]):
        """Validate structure of justification schema"""
        # Check if schema follows JSON Schema format
        if 'type' not in schema and 'properties' not in schema:
            self.validation_errors.append(OverrideJustificationError(
                error_code="OJ004",
                node_id=node.id,
                model_id=model_id,
                severity=OverrideValidationSeverity.ERROR.value,
                message="Justification schema not in valid JSON Schema format",
                expected="Valid JSON Schema with 'type' and 'properties'",
                actual="Invalid schema structure",
                autofix_available=True,
                suggested_schema=self._wrap_in_json_schema(schema)
            ))
        
        # Validate field types for core fields
        if 'properties' in schema:
            properties = schema['properties']
            
            # Validate 'who' field (should be string or object with user info)
            if 'who' in properties:
                who_field = properties['who']
                if who_field.get('type') not in ['string', 'object']:
                    self.validation_errors.append(OverrideJustificationError(
                        error_code="OJ005",
                        node_id=node.id,
                        model_id=model_id,
                        severity=OverrideValidationSeverity.ERROR.value,
                        message="'who' field should be string or object type",
                        expected="type: 'string' or 'object' for user identification",
                        actual=f"type: {who_field.get('type')}",
                        autofix_available=True
                    ))
            
            # Validate 'evidence' field (should be array or object)
            if 'evidence' in properties:
                evidence_field = properties['evidence']
                if evidence_field.get('type') not in ['array', 'object']:
                    self.validation_errors.append(OverrideJustificationError(
                        error_code="OJ006",
                        node_id=node.id,
                        model_id=model_id,
                        severity=OverrideValidationSeverity.ERROR.value,
                        message="'evidence' field should be array or object type",
                        expected="type: 'array' or 'object' for evidence pointers",
                        actual=f"type: {evidence_field.get('type')}",
                        autofix_available=True
                    ))
    
    def _validate_ledger_integration(self, node, model_id: Optional[str], schema: Dict[str, Any]):
        """Validate integration with override ledger"""
        # Check for ledger configuration
        ledger_config = self._extract_ledger_config(node)
        
        if not ledger_config:
            self.validation_errors.append(OverrideJustificationError(
                error_code="OJ007",
                node_id=node.id,
                model_id=model_id,
                severity=OverrideValidationSeverity.ERROR.value,
                message="Override node missing ledger integration configuration",
                expected="override_ledger_config specifying hash-chained recording",
                actual="No ledger configuration found",
                autofix_available=True,
                suggested_schema=self._generate_ledger_config()
            ))
        else:
            # Validate ledger configuration
            self._validate_ledger_config(node, model_id, ledger_config)
    
    def _extract_ledger_config(self, node) -> Optional[Dict[str, Any]]:
        """Extract override ledger configuration from node"""
        # Check direct parameter
        if 'override_ledger_config' in node.parameters:
            return node.parameters['override_ledger_config']
        
        # Check in override configuration
        override_config = node.parameters.get('override_config', {})
        if 'ledger_config' in override_config:
            return override_config['ledger_config']
        
        # Check metadata
        if 'ledger_config' in node.metadata:
            return node.metadata['ledger_config']
        
        return None
    
    def _validate_ledger_config(self, node, model_id: Optional[str], ledger_config: Dict[str, Any]):
        """Validate ledger configuration"""
        required_ledger_fields = ['enabled', 'hash_chaining', 'immutable_record']
        
        for field in required_ledger_fields:
            if field not in ledger_config:
                self.validation_errors.append(OverrideJustificationError(
                    error_code="OJ008",
                    node_id=node.id,
                    model_id=model_id,
                    severity=OverrideValidationSeverity.ERROR.value,
                    message=f"Ledger configuration missing required field: {field}",
                    expected=f"Required field '{field}' in ledger config",
                    actual="Field not found in ledger config",
                    autofix_available=True
                ))
        
        # Validate hash chaining is enabled
        if not ledger_config.get('hash_chaining', False):
            self.validation_errors.append(OverrideJustificationError(
                error_code="OJ009",
                node_id=node.id,
                model_id=model_id,
                severity=OverrideValidationSeverity.ERROR.value,
                message="Hash chaining not enabled for override ledger",
                expected="hash_chaining: true for audit integrity",
                actual=f"hash_chaining: {ledger_config.get('hash_chaining')}",
                autofix_available=True
            ))
    
    def _infer_override_type(self, node, model_id: Optional[str]) -> OverrideType:
        """Infer override type from node and model context"""
        # Check explicit override type
        override_type = node.parameters.get('override_type')
        if override_type:
            try:
                return OverrideType(override_type)
            except ValueError:
                pass
        
        # Infer from model ID
        if model_id:
            if any(pattern in model_id.lower() for pattern in ['fraud', 'credit', 'loan', 'financial']):
                return OverrideType.FINANCIAL_DECISION
            elif any(pattern in model_id.lower() for pattern in ['compliance', 'legal', 'regulatory']):
                return OverrideType.COMPLIANCE_OVERRIDE
            elif any(pattern in model_id.lower() for pattern in ['risk', 'security']):
                return OverrideType.RISK_ASSESSMENT
        
        # Default to manual review
        return OverrideType.MANUAL_REVIEW
    
    def _get_template_for_override_type(self, override_type: OverrideType) -> Optional[OverrideSchemaTemplate]:
        """Get schema template for override type"""
        template_mapping = {
            OverrideType.FINANCIAL_DECISION: "financial_override",
            OverrideType.RISK_ASSESSMENT: "risk_override",
            OverrideType.COMPLIANCE_OVERRIDE: "compliance_override",
            OverrideType.MANUAL_REVIEW: "manual_review",
            OverrideType.EMERGENCY_OVERRIDE: "emergency_override",
            OverrideType.POLICY_EXCEPTION: "policy_exception"
        }
        
        template_name = template_mapping.get(override_type)
        return self.schema_templates.get(template_name) if template_name else None
    
    def _generate_field_addition(self, schema: Dict[str, Any], field_name: str) -> Dict[str, Any]:
        """Generate schema with missing field added"""
        field_definitions = {
            'who': {
                'type': 'object',
                'properties': {
                    'user_id': {'type': 'string'},
                    'name': {'type': 'string'},
                    'role': {'type': 'string'},
                    'department': {'type': 'string'}
                },
                'required': ['user_id', 'name', 'role']
            },
            'why': {
                'type': 'string',
                'minLength': 10,
                'description': 'Detailed justification for the override'
            },
            'evidence': {
                'type': 'array',
                'items': {
                    'type': 'object',
                    'properties': {
                        'type': {'type': 'string', 'enum': ['document', 'approval', 'analysis', 'external_data']},
                        'reference': {'type': 'string'},
                        'description': {'type': 'string'}
                    }
                }
            },
            'timestamp': {
                'type': 'string',
                'format': 'date-time'
            },
            'override_type': {
                'type': 'string',
                'enum': ['financial_decision', 'risk_assessment', 'compliance_override', 'manual_review', 'emergency_override', 'policy_exception']
            }
        }
        
        updated_schema = schema.copy()
        if 'properties' not in updated_schema:
            updated_schema['properties'] = {}
        
        updated_schema['properties'][field_name] = field_definitions.get(field_name, {'type': 'string'})
        
        return updated_schema
    
    def _generate_audit_fields_addition(self, schema: Dict[str, Any], missing_fields: List[str]) -> Dict[str, Any]:
        """Generate schema with missing audit fields added"""
        audit_field_definitions = {
            'approver_id': {'type': 'string', 'description': 'ID of the approver'},
            'approval_timestamp': {'type': 'string', 'format': 'date-time'},
            'review_status': {'type': 'string', 'enum': ['pending', 'approved', 'rejected', 'escalated']}
        }
        
        updated_schema = schema.copy()
        if 'properties' not in updated_schema:
            updated_schema['properties'] = {}
        
        for field in missing_fields:
            if field in audit_field_definitions:
                updated_schema['properties'][field] = audit_field_definitions[field]
        
        return updated_schema
    
    def _wrap_in_json_schema(self, schema: Dict[str, Any]) -> Dict[str, Any]:
        """Wrap schema in proper JSON Schema format"""
        return {
            'type': 'object',
            'properties': schema,
            'required': ['who', 'why', 'evidence', 'timestamp'],
            'additionalProperties': False
        }
    
    def _generate_ledger_config(self) -> Dict[str, Any]:
        """Generate default ledger configuration"""
        return {
            'enabled': True,
            'hash_chaining': True,
            'immutable_record': True,
            'ledger_service': 'override_ledger',
            'integrity_verification': True,
            'audit_trail_retention_days': 2555  # 7 years for compliance
        }
    
    def _extract_model_id(self, node) -> Optional[str]:
        """Extract model_id from node"""
        return node.parameters.get('model_id') or node.metadata.get('model_id')
    
    def _init_schema_templates(self):
        """Initialize predefined schema templates"""
        # Financial decision override template
        self.schema_templates['financial_override'] = OverrideSchemaTemplate(
            template_name="financial_override",
            template_type="financial",
            description="Schema for financial decision overrides",
            required_fields=['who', 'why', 'evidence', 'financial_impact', 'risk_assessment'],
            optional_fields=['supervisor_approval', 'compliance_review'],
            schema={
                'type': 'object',
                'properties': {
                    'who': {
                        'type': 'object',
                        'properties': {
                            'user_id': {'type': 'string'},
                            'name': {'type': 'string'},
                            'role': {'type': 'string'},
                            'department': {'type': 'string'},
                            'authorization_level': {'type': 'string'}
                        },
                        'required': ['user_id', 'name', 'role', 'authorization_level']
                    },
                    'why': {
                        'type': 'string',
                        'minLength': 50,
                        'description': 'Detailed business justification'
                    },
                    'evidence': {
                        'type': 'array',
                        'items': {
                            'type': 'object',
                            'properties': {
                                'type': {'type': 'string', 'enum': ['financial_analysis', 'market_data', 'customer_communication', 'regulatory_guidance']},
                                'reference': {'type': 'string'},
                                'description': {'type': 'string'}
                            }
                        },
                        'minItems': 1
                    },
                    'financial_impact': {
                        'type': 'object',
                        'properties': {
                            'amount': {'type': 'number'},
                            'currency': {'type': 'string'},
                            'impact_type': {'type': 'string', 'enum': ['revenue', 'cost', 'risk']}
                        }
                    },
                    'risk_assessment': {
                        'type': 'string',
                        'enum': ['low', 'medium', 'high', 'critical']
                    },
                    'timestamp': {'type': 'string', 'format': 'date-time'},
                    'override_type': {'type': 'string', 'const': 'financial_decision'}
                },
                'required': ['who', 'why', 'evidence', 'financial_impact', 'risk_assessment', 'timestamp', 'override_type']
            },
            validation_rules=['financial_impact.amount must be documented', 'risk_assessment required for amounts > $10000']
        )
        
        # Compliance override template
        self.schema_templates['compliance_override'] = OverrideSchemaTemplate(
            template_name="compliance_override",
            template_type="compliance",
            description="Schema for compliance-related overrides",
            required_fields=['who', 'why', 'evidence', 'regulatory_basis', 'compliance_officer_approval'],
            optional_fields=['legal_review', 'external_counsel'],
            schema={
                'type': 'object',
                'properties': {
                    'who': {
                        'type': 'object',
                        'properties': {
                            'user_id': {'type': 'string'},
                            'name': {'type': 'string'},
                            'role': {'type': 'string'},
                            'compliance_certification': {'type': 'string'}
                        },
                        'required': ['user_id', 'name', 'role', 'compliance_certification']
                    },
                    'why': {
                        'type': 'string',
                        'minLength': 100,
                        'description': 'Detailed compliance justification'
                    },
                    'evidence': {
                        'type': 'array',
                        'items': {
                            'type': 'object',
                            'properties': {
                                'type': {'type': 'string', 'enum': ['regulatory_guidance', 'legal_opinion', 'audit_finding', 'policy_exception']},
                                'reference': {'type': 'string'},
                                'authority': {'type': 'string'}
                            }
                        }
                    },
                    'regulatory_basis': {
                        'type': 'string',
                        'description': 'Specific regulation or policy basis'
                    },
                    'compliance_officer_approval': {
                        'type': 'object',
                        'properties': {
                            'officer_id': {'type': 'string'},
                            'approval_date': {'type': 'string', 'format': 'date-time'},
                            'approval_reference': {'type': 'string'}
                        }
                    },
                    'timestamp': {'type': 'string', 'format': 'date-time'},
                    'override_type': {'type': 'string', 'const': 'compliance_override'}
                },
                'required': ['who', 'why', 'evidence', 'regulatory_basis', 'compliance_officer_approval', 'timestamp', 'override_type']
            },
            validation_rules=['compliance_officer_approval required', 'regulatory_basis must reference specific regulation']
        )
        
        # Manual review template
        self.schema_templates['manual_review'] = OverrideSchemaTemplate(
            template_name="manual_review",
            template_type="manual_review",
            description="Schema for manual review overrides",
            required_fields=['who', 'why', 'evidence', 'review_outcome'],
            optional_fields=['peer_review', 'supervisor_concurrence'],
            schema={
                'type': 'object',
                'properties': {
                    'who': {
                        'type': 'object',
                        'properties': {
                            'user_id': {'type': 'string'},
                            'name': {'type': 'string'},
                            'role': {'type': 'string'}
                        },
                        'required': ['user_id', 'name', 'role']
                    },
                    'why': {
                        'type': 'string',
                        'minLength': 20,
                        'description': 'Reason for manual override'
                    },
                    'evidence': {
                        'type': 'array',
                        'items': {
                            'type': 'object',
                            'properties': {
                                'type': {'type': 'string'},
                                'reference': {'type': 'string'},
                                'description': {'type': 'string'}
                            }
                        }
                    },
                    'review_outcome': {
                        'type': 'string',
                        'enum': ['approved', 'rejected', 'escalated', 'conditional']
                    },
                    'timestamp': {'type': 'string', 'format': 'date-time'},
                    'override_type': {'type': 'string', 'const': 'manual_review'}
                },
                'required': ['who', 'why', 'evidence', 'review_outcome', 'timestamp', 'override_type']
            },
            validation_rules=['review_outcome must be documented', 'evidence required for all overrides']
        )
    
    def generate_schema_template(self, override_type: str, custom_fields: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Generate JSON Schema template for override justification"""
        template = self.schema_templates.get(override_type)
        
        if not template:
            # Generate basic template
            base_schema = {
                'type': 'object',
                'properties': {
                    'who': {
                        'type': 'object',
                        'properties': {
                            'user_id': {'type': 'string'},
                            'name': {'type': 'string'},
                            'role': {'type': 'string'}
                        },
                        'required': ['user_id', 'name', 'role']
                    },
                    'why': {'type': 'string', 'minLength': 10},
                    'evidence': {
                        'type': 'array',
                        'items': {
                            'type': 'object',
                            'properties': {
                                'type': {'type': 'string'},
                                'reference': {'type': 'string'},
                                'description': {'type': 'string'}
                            }
                        }
                    },
                    'timestamp': {'type': 'string', 'format': 'date-time'},
                    'override_type': {'type': 'string', 'const': override_type}
                },
                'required': ['who', 'why', 'evidence', 'timestamp', 'override_type']
            }
            
            # Add custom fields if provided
            if custom_fields:
                base_schema['properties'].update(custom_fields)
            
            return base_schema
        
        # Return template schema with any custom additions
        schema = template.schema.copy()
        if custom_fields:
            schema['properties'].update(custom_fields)
        
        return schema
    
    def get_validation_summary(self) -> Dict[str, Any]:
        """Get summary of override justification validation"""
        error_count = len([e for e in self.validation_errors if e.severity == 'error'])
        warning_count = len([e for e in self.validation_errors if e.severity == 'warning'])
        info_count = len([e for e in self.validation_errors if e.severity == 'info'])
        
        return {
            'total_issues': len(self.validation_errors),
            'errors': error_count,
            'warnings': warning_count,
            'info': info_count,
            'validation_passed': error_count == 0,
            'autofix_available': len([e for e in self.validation_errors if e.autofix_available]),
            'issues_by_error_code': self._group_issues_by_error_code(),
            'common_violations': self._get_common_violations(),
            'available_templates': list(self.schema_templates.keys())
        }
    
    def _group_issues_by_error_code(self) -> Dict[str, int]:
        """Group issues by error code"""
        error_counts = {}
        for error in self.validation_errors:
            code = error.error_code
            if code not in error_counts:
                error_counts[code] = 0
            error_counts[code] += 1
        return error_counts
    
    def _get_common_violations(self) -> List[str]:
        """Get most common validation violations"""
        violation_descriptions = {
            'OJ001': 'Missing justification schema',
            'OJ002': 'Missing required fields',
            'OJ003': 'Missing audit fields',
            'OJ004': 'Invalid schema structure',
            'OJ005': 'Invalid who field type',
            'OJ006': 'Invalid evidence field type',
            'OJ007': 'Missing ledger integration',
            'OJ008': 'Missing ledger config fields',
            'OJ009': 'Hash chaining not enabled'
        }
        
        error_counts = self._group_issues_by_error_code()
        sorted_violations = sorted(error_counts.items(), key=lambda x: x[1], reverse=True)
        
        return [f"{violation_descriptions.get(code, code)}: {count} occurrences" 
                for code, count in sorted_violations[:5]]

# Test helper functions
def create_test_nodes_for_override_validation():
    """Create test nodes with various override validation scenarios"""
    from .ir import IRNode, IRNodeType
    
    nodes = [
        # Node with proper override configuration
        IRNode(
            id="proper_override",
            type=IRNodeType.ML_PREDICT,
            parameters={
                'model_id': 'financial_scorer_v1',
                'override_allowed': True,
                'override_justification_schema': {
                    'type': 'object',
                    'properties': {
                        'who': {'type': 'object', 'properties': {'user_id': {'type': 'string'}, 'name': {'type': 'string'}, 'role': {'type': 'string'}}},
                        'why': {'type': 'string'},
                        'evidence': {'type': 'array'},
                        'timestamp': {'type': 'string', 'format': 'date-time'},
                        'override_type': {'type': 'string'}
                    }
                },
                'override_ledger_config': {
                    'enabled': True,
                    'hash_chaining': True,
                    'immutable_record': True
                }
            }
        ),
        
        # Node with override allowed but missing schema
        IRNode(
            id="missing_schema",
            type=IRNodeType.ML_SCORE,
            parameters={
                'model_id': 'credit_scorer_v2',
                'override_allowed': True
                # Missing justification schema
            }
        ),
        
        # Node with incomplete schema
        IRNode(
            id="incomplete_schema",
            type=IRNodeType.ML_CLASSIFY,
            parameters={
                'model_id': 'fraud_detector_v1',
                'override_allowed': True,
                'override_justification_schema': {
                    'who': {'type': 'string'},  # Missing other required fields
                    'why': {'type': 'string'}
                }
            }
        ),
        
        # Node with schema but missing ledger config
        IRNode(
            id="missing_ledger",
            type=IRNodeType.ML_PREDICT,
            parameters={
                'model_id': 'compliance_checker_v1',
                'override_allowed': True,
                'override_justification_schema': {
                    'type': 'object',
                    'properties': {
                        'who': {'type': 'string'},
                        'why': {'type': 'string'},
                        'evidence': {'type': 'array'},
                        'timestamp': {'type': 'string'}
                    }
                }
                # Missing ledger config
            }
        )
    ]
    
    return nodes

def run_override_justification_validation_tests():
    """Run override justification validation tests"""
    validator = OverrideJustificationValidator()
    
    # Create test graph
    test_nodes = create_test_nodes_for_override_validation()
    from .ir import IRGraph
    
    test_graph = IRGraph(
        workflow_id="test_override_validation",
        name="Test Override Validation Workflow",
        version="1.0.0",
        automation_type="rbia",
        nodes=test_nodes
    )
    
    # Run validation
    errors = validator.validate_override_justification_schemas(test_graph)
    
    print(f"Override justification validation completed. Found {len(errors)} issues:")
    for error in errors:
        print(f"  {error.error_code} [{error.severity}] {error.node_id}: {error.message}")
        if error.autofix_available:
            print(f"    Autofix available: {error.autofix_available}")
    
    # Print summary
    summary = validator.get_validation_summary()
    print(f"\nValidation Summary:")
    print(f"  Validation passed: {summary['validation_passed']}")
    print(f"  Errors: {summary['errors']}")
    print(f"  Warnings: {summary['warnings']}")
    print(f"  Autofix available: {summary['autofix_available']}")
    
    # Test schema template generation
    print(f"\nTesting schema template generation:")
    financial_template = validator.generate_schema_template('financial_override')
    print(f"  Generated financial override template with {len(financial_template['properties'])} fields")
    
    return errors, summary

if __name__ == "__main__":
    run_override_justification_validation_tests()
