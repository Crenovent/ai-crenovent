"""
SoD (Segregation of Duties) Validator - Task 6.2.23
====================================================

SoD checks: same persona cannot author+approve ML change
- Analyzer/CI gate that reads author and approver metadata from manifest fields
- Validates against IAM mapping (role → persona mapping)
- Fails compile or flags for manual CAB approval if SoD violation occurs
- Provides exception workflow with audit trail if necessary

Dependencies: Task 6.2.4 (IR), IAM service integration
Outputs: SoD validation → ensures governance compliance at compile time
"""

from typing import Dict, List, Any, Optional, Set, Tuple
from dataclasses import dataclass, field
from enum import Enum
import logging
from datetime import datetime

logger = logging.getLogger(__name__)

@dataclass
class SoDViolation:
    """SoD (Segregation of Duties) violation"""
    violation_code: str
    node_id: str
    model_id: Optional[str]
    severity: str  # "error", "warning", "info"
    violation_type: str  # "author_approver_same", "insufficient_approvers", "role_conflict"
    message: str
    author_info: Dict[str, Any]
    approver_info: List[Dict[str, Any]]
    expected: str
    actual: str
    requires_cab_approval: bool = False
    exception_available: bool = False
    audit_trail_required: bool = True

@dataclass
class PersonaMapping:
    """Persona mapping from IAM service"""
    user_id: str
    persona_id: str
    roles: List[str]
    department: str
    authorization_level: str
    can_author: bool
    can_approve: bool
    approval_limits: Dict[str, Any]

@dataclass
class SoDException:
    """SoD exception with audit trail"""
    exception_id: str
    violation_code: str
    node_id: str
    justification: str
    approved_by: str
    approval_date: datetime
    exception_type: str  # "emergency", "business_critical", "regulatory_required"
    expiration_date: Optional[datetime] = None
    audit_references: List[str] = field(default_factory=list)

class SoDValidationSeverity(Enum):
    """Severity levels for SoD validation"""
    ERROR = "error"
    WARNING = "warning"
    INFO = "info"

class SoDViolationType(Enum):
    """Types of SoD violations"""
    AUTHOR_APPROVER_SAME = "author_approver_same"
    INSUFFICIENT_APPROVERS = "insufficient_approvers"
    ROLE_CONFLICT = "role_conflict"
    PERSONA_CONFLICT = "persona_conflict"
    AUTHORIZATION_EXCEEDED = "authorization_exceeded"

# Task 6.2.23: SoD (Segregation of Duties) Validator
class SoDValidator:
    """Validates Segregation of Duties for ML workflow changes"""
    
    def __init__(self, iam_service_url: Optional[str] = None):
        self.sod_violations: List[SoDViolation] = []
        self.persona_mappings: Dict[str, PersonaMapping] = {}
        self.sod_exceptions: Dict[str, SoDException] = {}
        self.iam_service_url = iam_service_url
        
        # Initialize mock IAM data for testing
        self._init_mock_iam_data()
    
    def validate_sod_compliance(self, ir_graph, manifest_metadata: Dict[str, Any]) -> List[SoDViolation]:
        """Validate SoD compliance for workflow changes"""
        from .ir import IRGraph, IRNodeType
        
        self.sod_violations.clear()
        
        # Extract author and approver information
        author_info = self._extract_author_info(manifest_metadata)
        approver_info = self._extract_approver_info(manifest_metadata)
        
        if not author_info:
            self._report_missing_author_info(ir_graph.workflow_id)
            return self.sod_violations
        
        # Validate ML nodes for SoD compliance
        ml_nodes = ir_graph.get_ml_nodes()
        for node in ml_nodes:
            self._validate_node_sod_compliance(node, author_info, approver_info)
        
        # Validate workflow-level SoD compliance
        self._validate_workflow_sod_compliance(ir_graph, author_info, approver_info)
        
        return self.sod_violations
    
    def _extract_author_info(self, manifest_metadata: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Extract author information from manifest metadata"""
        # Check various possible locations for author info
        author_fields = ['author_id', 'created_by', 'workflow_author', 'change_author']
        
        for field in author_fields:
            if field in manifest_metadata:
                author_id = manifest_metadata[field]
                return {
                    'user_id': author_id,
                    'field_source': field,
                    'persona_mapping': self._get_persona_mapping(author_id),
                    'timestamp': manifest_metadata.get('created_at', datetime.utcnow().isoformat())
                }
        
        # Check nested structures
        if 'provenance' in manifest_metadata:
            provenance = manifest_metadata['provenance']
            if 'author' in provenance:
                author_data = provenance['author']
                return {
                    'user_id': author_data.get('user_id') or author_data.get('id'),
                    'field_source': 'provenance.author',
                    'persona_mapping': self._get_persona_mapping(author_data.get('user_id') or author_data.get('id')),
                    'name': author_data.get('name'),
                    'role': author_data.get('role'),
                    'timestamp': author_data.get('timestamp')
                }
        
        return None
    
    def _extract_approver_info(self, manifest_metadata: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Extract approver information from manifest metadata"""
        approvers = []
        
        # Check various possible locations for approver info
        approver_fields = ['approver_id', 'approved_by', 'reviewers', 'approvers']
        
        for field in approver_fields:
            if field in manifest_metadata:
                approver_data = manifest_metadata[field]
                
                # Handle single approver
                if isinstance(approver_data, str):
                    approvers.append({
                        'user_id': approver_data,
                        'field_source': field,
                        'persona_mapping': self._get_persona_mapping(approver_data),
                        'timestamp': manifest_metadata.get('approved_at', datetime.utcnow().isoformat())
                    })
                
                # Handle multiple approvers
                elif isinstance(approver_data, list):
                    for approver in approver_data:
                        if isinstance(approver, str):
                            approvers.append({
                                'user_id': approver,
                                'field_source': field,
                                'persona_mapping': self._get_persona_mapping(approver)
                            })
                        elif isinstance(approver, dict):
                            approvers.append({
                                'user_id': approver.get('user_id') or approver.get('id'),
                                'field_source': field,
                                'persona_mapping': self._get_persona_mapping(approver.get('user_id') or approver.get('id')),
                                'name': approver.get('name'),
                                'role': approver.get('role'),
                                'timestamp': approver.get('timestamp')
                            })
        
        # Check approval workflow structure
        if 'approval_workflow' in manifest_metadata:
            approval_workflow = manifest_metadata['approval_workflow']
            if 'approvals' in approval_workflow:
                for approval in approval_workflow['approvals']:
                    approvers.append({
                        'user_id': approval.get('approver_id'),
                        'field_source': 'approval_workflow',
                        'persona_mapping': self._get_persona_mapping(approval.get('approver_id')),
                        'approval_status': approval.get('status'),
                        'approval_timestamp': approval.get('timestamp')
                    })
        
        return approvers
    
    def _validate_node_sod_compliance(self, node, author_info: Dict[str, Any], approver_info: List[Dict[str, Any]]):
        """Validate SoD compliance for a single ML node"""
        model_id = self._extract_model_id(node)
        
        # Check if node requires SoD validation
        if not self._requires_sod_validation(node):
            return
        
        # Get SoD policy for this node
        sod_policy = self._get_node_sod_policy(node)
        
        # Validate author-approver separation
        self._validate_author_approver_separation(node, model_id, author_info, approver_info, sod_policy)
        
        # Validate sufficient approvers
        self._validate_sufficient_approvers(node, model_id, approver_info, sod_policy)
        
        # Validate role conflicts
        self._validate_role_conflicts(node, model_id, author_info, approver_info, sod_policy)
        
        # Validate authorization levels
        self._validate_authorization_levels(node, model_id, author_info, approver_info, sod_policy)
    
    def _validate_workflow_sod_compliance(self, ir_graph, author_info: Dict[str, Any], approver_info: List[Dict[str, Any]]):
        """Validate workflow-level SoD compliance"""
        # Check for high-risk workflow patterns
        ml_node_count = len(ir_graph.get_ml_nodes())
        
        if ml_node_count > 5:  # High complexity workflow
            # Require additional approver for complex workflows
            if len(approver_info) < 2:
                self.sod_violations.append(SoDViolation(
                    violation_code="SOD001",
                    node_id=ir_graph.workflow_id,
                    model_id=None,
                    severity=SoDValidationSeverity.ERROR.value,
                    violation_type=SoDViolationType.INSUFFICIENT_APPROVERS.value,
                    message=f"Complex workflow with {ml_node_count} ML nodes requires multiple approvers",
                    author_info=author_info,
                    approver_info=approver_info,
                    expected="At least 2 approvers for complex workflows",
                    actual=f"{len(approver_info)} approver(s)",
                    requires_cab_approval=True
                ))
    
    def _validate_author_approver_separation(self, node, model_id: Optional[str], 
                                           author_info: Dict[str, Any], 
                                           approver_info: List[Dict[str, Any]], 
                                           sod_policy: Dict[str, Any]):
        """Validate that author and approver are different personas"""
        if not sod_policy.get('author_approver_separation', True):
            return  # SoD not required for this node
        
        author_persona = self._get_persona_from_info(author_info)
        
        for approver in approver_info:
            approver_persona = self._get_persona_from_info(approver)
            
            # Check if same person
            if author_info['user_id'] == approver['user_id']:
                self.sod_violations.append(SoDViolation(
                    violation_code="SOD002",
                    node_id=node.id,
                    model_id=model_id,
                    severity=SoDValidationSeverity.ERROR.value,
                    violation_type=SoDViolationType.AUTHOR_APPROVER_SAME.value,
                    message="Author and approver cannot be the same person",
                    author_info=author_info,
                    approver_info=[approver],
                    expected="Different person for author and approver",
                    actual=f"Same person: {author_info['user_id']}",
                    requires_cab_approval=True,
                    exception_available=True
                ))
            
            # Check if same persona (even if different people)
            elif author_persona and approver_persona and author_persona['persona_id'] == approver_persona['persona_id']:
                self.sod_violations.append(SoDViolation(
                    violation_code="SOD003",
                    node_id=node.id,
                    model_id=model_id,
                    severity=SoDValidationSeverity.WARNING.value,
                    violation_type=SoDViolationType.PERSONA_CONFLICT.value,
                    message="Author and approver have same persona/role",
                    author_info=author_info,
                    approver_info=[approver],
                    expected="Different personas for author and approver",
                    actual=f"Same persona: {author_persona['persona_id']}",
                    requires_cab_approval=False,
                    exception_available=True
                ))
    
    def _validate_sufficient_approvers(self, node, model_id: Optional[str], 
                                     approver_info: List[Dict[str, Any]], 
                                     sod_policy: Dict[str, Any]):
        """Validate sufficient number of approvers"""
        required_approvers = sod_policy.get('required_approvers', 1)
        actual_approvers = len(approver_info)
        
        if actual_approvers < required_approvers:
            self.sod_violations.append(SoDViolation(
                violation_code="SOD004",
                node_id=node.id,
                model_id=model_id,
                severity=SoDValidationSeverity.ERROR.value,
                violation_type=SoDViolationType.INSUFFICIENT_APPROVERS.value,
                message=f"Insufficient approvers: {actual_approvers} of {required_approvers} required",
                author_info={},
                approver_info=approver_info,
                expected=f"{required_approvers} approver(s)",
                actual=f"{actual_approvers} approver(s)",
                requires_cab_approval=True
            ))
    
    def _validate_role_conflicts(self, node, model_id: Optional[str], 
                               author_info: Dict[str, Any], 
                               approver_info: List[Dict[str, Any]], 
                               sod_policy: Dict[str, Any]):
        """Validate no conflicting roles in approval chain"""
        excluded_combinations = sod_policy.get('excluded_combinations', [])
        
        author_roles = self._get_roles_from_info(author_info)
        
        for approver in approver_info:
            approver_roles = self._get_roles_from_info(approver)
            
            # Check for excluded role combinations
            for author_role in author_roles:
                for approver_role in approver_roles:
                    if [author_role, approver_role] in excluded_combinations or \
                       [approver_role, author_role] in excluded_combinations:
                        self.sod_violations.append(SoDViolation(
                            violation_code="SOD005",
                            node_id=node.id,
                            model_id=model_id,
                            severity=SoDValidationSeverity.ERROR.value,
                            violation_type=SoDViolationType.ROLE_CONFLICT.value,
                            message=f"Conflicting roles: {author_role} (author) and {approver_role} (approver)",
                            author_info=author_info,
                            approver_info=[approver],
                            expected="Non-conflicting role combination",
                            actual=f"Excluded combination: {author_role} + {approver_role}",
                            requires_cab_approval=True
                        ))
    
    def _validate_authorization_levels(self, node, model_id: Optional[str], 
                                     author_info: Dict[str, Any], 
                                     approver_info: List[Dict[str, Any]], 
                                     sod_policy: Dict[str, Any]):
        """Validate authorization levels for high-risk changes"""
        if not self._is_high_risk_change(node, model_id):
            return
        
        # Check if approvers have sufficient authorization
        required_auth_level = sod_policy.get('min_authorization_level', 'senior')
        
        for approver in approver_info:
            approver_persona = self._get_persona_from_info(approver)
            if approver_persona:
                auth_level = approver_persona.authorization_level
                if not self._has_sufficient_authorization(auth_level, required_auth_level):
                    self.sod_violations.append(SoDViolation(
                        violation_code="SOD006",
                        node_id=node.id,
                        model_id=model_id,
                        severity=SoDValidationSeverity.ERROR.value,
                        violation_type=SoDViolationType.AUTHORIZATION_EXCEEDED.value,
                        message=f"Approver lacks sufficient authorization: {auth_level} < {required_auth_level}",
                        author_info=author_info,
                        approver_info=[approver],
                        expected=f"Authorization level >= {required_auth_level}",
                        actual=f"Authorization level: {auth_level}",
                        requires_cab_approval=True
                    ))
    
    def _requires_sod_validation(self, node) -> bool:
        """Check if node requires SoD validation"""
        # Always validate ML nodes
        from .ir import IRNodeType
        ml_types = {IRNodeType.ML_NODE, IRNodeType.ML_PREDICT, 
                   IRNodeType.ML_SCORE, IRNodeType.ML_CLASSIFY, IRNodeType.ML_EXPLAIN}
        if node.type in ml_types:
            return True
        
        # Check if node has SoD policy
        for policy in node.policies:
            if policy.policy_type == 'sod':
                return True
        
        # Check if node is marked as requiring SoD
        if node.parameters.get('sod_required', False):
            return True
        
        return False
    
    def _get_node_sod_policy(self, node) -> Dict[str, Any]:
        """Get SoD policy for node"""
        # Check node-specific SoD policies
        for policy in node.policies:
            if policy.policy_type == 'sod':
                return policy.parameters
        
        # Return default SoD policy
        return {
            'author_approver_separation': True,
            'required_approvers': 1,
            'excluded_combinations': [['author', 'primary_approver']],
            'min_authorization_level': 'standard'
        }
    
    def _get_persona_mapping(self, user_id: str) -> Optional[PersonaMapping]:
        """Get persona mapping for user from IAM service"""
        if user_id in self.persona_mappings:
            return self.persona_mappings[user_id]
        
        # In real implementation, this would call IAM service
        # For now, return mock data
        return self._get_mock_persona_mapping(user_id)
    
    def _get_persona_from_info(self, user_info: Dict[str, Any]) -> Optional[PersonaMapping]:
        """Get persona mapping from user info"""
        user_id = user_info.get('user_id')
        if user_id:
            return user_info.get('persona_mapping') or self._get_persona_mapping(user_id)
        return None
    
    def _get_roles_from_info(self, user_info: Dict[str, Any]) -> List[str]:
        """Get roles from user info"""
        persona = self._get_persona_from_info(user_info)
        if persona:
            return persona.roles
        
        # Fallback to direct role info
        if 'role' in user_info:
            return [user_info['role']]
        
        return []
    
    def _is_high_risk_change(self, node, model_id: Optional[str]) -> bool:
        """Check if node represents high-risk change"""
        # Check explicit risk level
        if node.parameters.get('risk_level') == 'high':
            return True
        
        # Check model ID patterns
        if model_id:
            high_risk_patterns = ['fraud', 'credit', 'loan', 'financial', 'compliance', 'legal']
            if any(pattern in model_id.lower() for pattern in high_risk_patterns):
                return True
        
        return False
    
    def _has_sufficient_authorization(self, actual_level: str, required_level: str) -> bool:
        """Check if authorization level is sufficient"""
        auth_hierarchy = ['junior', 'standard', 'senior', 'principal', 'director', 'vp']
        
        try:
            actual_index = auth_hierarchy.index(actual_level.lower())
            required_index = auth_hierarchy.index(required_level.lower())
            return actual_index >= required_index
        except ValueError:
            # Unknown authorization level
            return False
    
    def _extract_model_id(self, node) -> Optional[str]:
        """Extract model_id from node"""
        return node.parameters.get('model_id') or node.metadata.get('model_id')
    
    def _report_missing_author_info(self, workflow_id: str):
        """Report missing author information"""
        self.sod_violations.append(SoDViolation(
            violation_code="SOD000",
            node_id=workflow_id,
            model_id=None,
            severity=SoDValidationSeverity.ERROR.value,
            violation_type="missing_author",
            message="Missing author information in manifest metadata",
            author_info={},
            approver_info=[],
            expected="Author metadata (author_id, created_by, etc.)",
            actual="No author information found",
            requires_cab_approval=True
        ))
    
    def _init_mock_iam_data(self):
        """Initialize mock IAM data for testing"""
        # Mock persona mappings
        self.persona_mappings = {
            'john.doe': PersonaMapping(
                user_id='john.doe',
                persona_id='ml_engineer',
                roles=['ml_engineer', 'data_scientist'],
                department='engineering',
                authorization_level='senior',
                can_author=True,
                can_approve=True,
                approval_limits={'financial_impact': 50000}
            ),
            'jane.smith': PersonaMapping(
                user_id='jane.smith',
                persona_id='ml_engineer',
                roles=['ml_engineer'],
                department='engineering',
                authorization_level='standard',
                can_author=True,
                can_approve=False,
                approval_limits={}
            ),
            'bob.wilson': PersonaMapping(
                user_id='bob.wilson',
                persona_id='compliance_officer',
                roles=['compliance_officer', 'auditor'],
                department='compliance',
                authorization_level='principal',
                can_author=False,
                can_approve=True,
                approval_limits={'financial_impact': 1000000}
            ),
            'alice.johnson': PersonaMapping(
                user_id='alice.johnson',
                persona_id='director',
                roles=['engineering_director', 'approver'],
                department='engineering',
                authorization_level='director',
                can_author=False,
                can_approve=True,
                approval_limits={'financial_impact': 5000000}
            )
        }
    
    def _get_mock_persona_mapping(self, user_id: str) -> Optional[PersonaMapping]:
        """Get mock persona mapping for testing"""
        # Return default mapping for unknown users
        return PersonaMapping(
            user_id=user_id,
            persona_id='unknown',
            roles=['user'],
            department='unknown',
            authorization_level='standard',
            can_author=True,
            can_approve=False,
            approval_limits={}
        )
    
    def create_sod_exception(self, violation_code: str, node_id: str, 
                           justification: str, approved_by: str, 
                           exception_type: str = "business_critical") -> SoDException:
        """Create SoD exception with audit trail"""
        exception = SoDException(
            exception_id=f"SOD_EXC_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}",
            violation_code=violation_code,
            node_id=node_id,
            justification=justification,
            approved_by=approved_by,
            approval_date=datetime.utcnow(),
            exception_type=exception_type,
            audit_references=[f"approval_{approved_by}_{datetime.utcnow().isoformat()}"]
        )
        
        self.sod_exceptions[exception.exception_id] = exception
        return exception
    
    def is_exception_valid(self, violation_code: str, node_id: str) -> bool:
        """Check if valid SoD exception exists"""
        for exception in self.sod_exceptions.values():
            if (exception.violation_code == violation_code and 
                exception.node_id == node_id and
                (not exception.expiration_date or exception.expiration_date > datetime.utcnow())):
                return True
        return False
    
    def get_validation_summary(self) -> Dict[str, Any]:
        """Get summary of SoD validation results"""
        error_count = len([v for v in self.sod_violations if v.severity == 'error'])
        warning_count = len([v for v in self.sod_violations if v.severity == 'warning'])
        info_count = len([v for v in self.sod_violations if v.severity == 'info'])
        
        cab_required = len([v for v in self.sod_violations if v.requires_cab_approval])
        exceptions_available = len([v for v in self.sod_violations if v.exception_available])
        
        return {
            'total_violations': len(self.sod_violations),
            'errors': error_count,
            'warnings': warning_count,
            'info': info_count,
            'sod_compliance_passed': error_count == 0,
            'cab_approval_required': cab_required,
            'exceptions_available': exceptions_available,
            'violations_by_type': self._group_violations_by_type(),
            'violations_by_node': self._group_violations_by_node(),
            'active_exceptions': len(self.sod_exceptions),
            'common_violations': self._get_common_violations()
        }
    
    def _group_violations_by_type(self) -> Dict[str, int]:
        """Group violations by type"""
        type_counts = {}
        for violation in self.sod_violations:
            vtype = violation.violation_type
            if vtype not in type_counts:
                type_counts[vtype] = 0
            type_counts[vtype] += 1
        return type_counts
    
    def _group_violations_by_node(self) -> Dict[str, int]:
        """Group violations by node"""
        node_counts = {}
        for violation in self.sod_violations:
            node_id = violation.node_id
            if node_id not in node_counts:
                node_counts[node_id] = 0
            node_counts[node_id] += 1
        return node_counts
    
    def _get_common_violations(self) -> List[str]:
        """Get most common SoD violations"""
        violation_descriptions = {
            'SOD001': 'Complex workflow insufficient approvers',
            'SOD002': 'Author and approver same person',
            'SOD003': 'Author and approver same persona',
            'SOD004': 'Insufficient approvers',
            'SOD005': 'Conflicting roles',
            'SOD006': 'Insufficient authorization level'
        }
        
        violation_counts = {}
        for violation in self.sod_violations:
            code = violation.violation_code
            if code not in violation_counts:
                violation_counts[code] = 0
            violation_counts[code] += 1
        
        sorted_violations = sorted(violation_counts.items(), key=lambda x: x[1], reverse=True)
        
        return [f"{violation_descriptions.get(code, code)}: {count} occurrences" 
                for code, count in sorted_violations[:5]]

# Test helper functions
def create_test_manifest_with_sod_issues():
    """Create test manifest metadata with SoD issues"""
    return {
        'workflow_id': 'test_sod_workflow',
        'version': '1.0.0',
        'author_id': 'john.doe',  # Same as approver - SoD violation
        'created_by': 'john.doe',
        'created_at': '2024-01-15T10:00:00Z',
        'approver_id': 'john.doe',  # Same as author - violation
        'approved_by': 'john.doe',
        'approved_at': '2024-01-15T11:00:00Z',
        'provenance': {
            'author': {
                'user_id': 'john.doe',
                'name': 'John Doe',
                'role': 'ml_engineer'
            }
        },
        'approval_workflow': {
            'approvals': [
                {
                    'approver_id': 'john.doe',  # Same person again
                    'status': 'approved',
                    'timestamp': '2024-01-15T11:00:00Z'
                }
            ]
        }
    }

def create_test_nodes_for_sod_validation():
    """Create test ML nodes for SoD validation"""
    from .ir import IRNode, IRNodeType, IRPolicy
    
    nodes = [
        # High-risk financial node
        IRNode(
            id="financial_decision",
            type=IRNodeType.ML_PREDICT,
            parameters={
                'model_id': 'financial_risk_scorer_v1',
                'risk_level': 'high',
                'sod_required': True
            },
            policies=[
                IRPolicy(
                    policy_id="sod_policy_financial",
                    policy_type="sod",
                    parameters={
                        'author_approver_separation': True,
                        'required_approvers': 2,
                        'excluded_combinations': [['ml_engineer', 'ml_engineer']],
                        'min_authorization_level': 'senior'
                    }
                )
            ]
        ),
        
        # Standard ML node
        IRNode(
            id="standard_ml_node",
            type=IRNodeType.ML_SCORE,
            parameters={
                'model_id': 'churn_predictor_v2'
            }
        ),
        
        # Compliance-related node
        IRNode(
            id="compliance_check",
            type=IRNodeType.ML_CLASSIFY,
            parameters={
                'model_id': 'compliance_checker_v1',
                'risk_level': 'high'
            }
        )
    ]
    
    return nodes

def run_sod_validation_tests():
    """Run SoD validation tests"""
    validator = SoDValidator()
    
    # Create test graph and manifest
    test_nodes = create_test_nodes_for_sod_validation()
    test_manifest = create_test_manifest_with_sod_issues()
    
    from .ir import IRGraph
    
    test_graph = IRGraph(
        workflow_id="test_sod_validation",
        name="Test SoD Validation Workflow",
        version="1.0.0",
        automation_type="rbia",
        nodes=test_nodes
    )
    
    # Run validation
    violations = validator.validate_sod_compliance(test_graph, test_manifest)
    
    print(f"SoD validation completed. Found {len(violations)} violations:")
    for violation in violations:
        print(f"  {violation.violation_code} [{violation.severity}] {violation.node_id}: {violation.message}")
        if violation.requires_cab_approval:
            print(f"    → Requires CAB approval")
        if violation.exception_available:
            print(f"    → Exception workflow available")
    
    # Print summary
    summary = validator.get_validation_summary()
    print(f"\nValidation Summary:")
    print(f"  SoD compliance passed: {summary['sod_compliance_passed']}")
    print(f"  Errors: {summary['errors']}")
    print(f"  Warnings: {summary['warnings']}")
    print(f"  CAB approval required: {summary['cab_approval_required']}")
    print(f"  Exceptions available: {summary['exceptions_available']}")
    
    # Test exception creation
    if violations:
        first_violation = violations[0]
        exception = validator.create_sod_exception(
            first_violation.violation_code,
            first_violation.node_id,
            "Emergency business requirement - approved by director",
            "alice.johnson",
            "emergency"
        )
        print(f"\nCreated SoD exception: {exception.exception_id}")
    
    return violations, summary

if __name__ == "__main__":
    run_sod_validation_tests()
