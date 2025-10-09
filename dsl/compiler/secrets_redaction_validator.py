"""
Secrets Redaction Validator - Task 6.2.25
==========================================

Secrets redaction: ensure plan has no secrets
- Linter that scans IR and metadata for plaintext secrets patterns (regex for keys, tokens, certs)
- Checks for disallowed fields or accidentally embedded credentials
- Enforces that external credentials must be referenced by secure URIs (vault://secret/path#key)
- Provides autofix suggestions: replace inline secret with vault reference and fail CI if secret remains

Dependencies: Task 6.2.4 (IR)
Outputs: Secrets redaction validation → prevents credential leakage in plan manifests
"""

from typing import Dict, List, Any, Optional, Set, Tuple, Pattern
from dataclasses import dataclass, field
from enum import Enum
import logging
import re
import base64
from datetime import datetime

logger = logging.getLogger(__name__)

@dataclass
class SecretViolation:
    """Secret exposure violation"""
    violation_code: str
    node_id: Optional[str]
    field_path: str
    secret_type: str  # "api_key", "password", "token", "certificate", "connection_string"
    severity: str  # "error", "warning", "info"
    message: str
    detected_pattern: str
    masked_value: str  # Partially masked for reporting
    expected: str
    actual: str
    autofix_available: bool = False
    suggested_vault_reference: Optional[str] = None
    remediation_steps: List[str] = field(default_factory=list)

@dataclass
class SecretPattern:
    """Pattern for detecting secrets"""
    name: str
    pattern: Pattern[str]
    secret_type: str
    confidence: str  # "high", "medium", "low"
    description: str
    vault_prefix: str  # Suggested vault path prefix

@dataclass
class VaultReference:
    """Secure vault reference"""
    vault_uri: str
    secret_path: str
    key_name: str
    description: str
    access_policy: str

class SecretValidationSeverity(Enum):
    """Severity levels for secret validation"""
    ERROR = "error"
    WARNING = "warning"
    INFO = "info"

class SecretType(Enum):
    """Types of secrets to detect"""
    API_KEY = "api_key"
    PASSWORD = "password"
    TOKEN = "token"
    CERTIFICATE = "certificate"
    CONNECTION_STRING = "connection_string"
    PRIVATE_KEY = "private_key"
    OAUTH_SECRET = "oauth_secret"
    DATABASE_URL = "database_url"

# Task 6.2.25: Secrets Redaction Validator
class SecretsRedactionValidator:
    """Validates that plan manifests contain no exposed secrets"""
    
    def __init__(self):
        self.secret_violations: List[SecretViolation] = []
        self.secret_patterns: List[SecretPattern] = []
        self.disallowed_fields: Set[str] = set()
        self.vault_references: Dict[str, VaultReference] = {}
        
        # Initialize secret detection patterns
        self._init_secret_patterns()
        self._init_disallowed_fields()
    
    def validate_secrets_redaction(self, ir_graph, manifest_metadata: Optional[Dict[str, Any]] = None) -> List[SecretViolation]:
        """Validate that no secrets are exposed in IR graph or manifest"""
        from .ir import IRGraph
        
        self.secret_violations.clear()
        
        # Validate IR graph nodes
        for node in ir_graph.nodes:
            self._validate_node_secrets(node)
        
        # Validate IR graph metadata
        self._validate_metadata_secrets(ir_graph.metadata, "ir_graph.metadata")
        
        # Validate governance configuration
        if ir_graph.governance:
            self._validate_metadata_secrets(ir_graph.governance, "ir_graph.governance")
        
        # Validate manifest metadata if provided
        if manifest_metadata:
            self._validate_metadata_secrets(manifest_metadata, "manifest")
        
        return self.secret_violations
    
    def _validate_node_secrets(self, node):
        """Validate secrets in a single node"""
        # Validate node parameters
        self._scan_dict_for_secrets(node.parameters, f"node.{node.id}.parameters", node.id)
        
        # Validate node metadata
        self._scan_dict_for_secrets(node.metadata, f"node.{node.id}.metadata", node.id)
        
        # Validate policy parameters
        for i, policy in enumerate(node.policies):
            self._scan_dict_for_secrets(
                policy.parameters, 
                f"node.{node.id}.policies[{i}].parameters", 
                node.id
            )
        
        # Validate input/output configurations
        for i, input_spec in enumerate(node.inputs):
            if hasattr(input_spec, 'validation') and input_spec.validation:
                self._scan_dict_for_secrets(
                    input_spec.validation,
                    f"node.{node.id}.inputs[{i}].validation",
                    node.id
                )
        
        for i, output_spec in enumerate(node.outputs):
            if hasattr(output_spec, 'schema') and output_spec.schema:
                self._scan_dict_for_secrets(
                    output_spec.schema,
                    f"node.{node.id}.outputs[{i}].schema",
                    node.id
                )
    
    def _validate_metadata_secrets(self, metadata: Dict[str, Any], base_path: str):
        """Validate secrets in metadata dictionary"""
        self._scan_dict_for_secrets(metadata, base_path, None)
    
    def _scan_dict_for_secrets(self, data: Any, path: str, node_id: Optional[str]):
        """Recursively scan dictionary for secrets"""
        if isinstance(data, dict):
            for key, value in data.items():
                current_path = f"{path}.{key}"
                
                # Check if field is disallowed
                if key.lower() in self.disallowed_fields:
                    self._report_disallowed_field(key, current_path, value, node_id)
                
                # Scan value
                self._scan_value_for_secrets(key, value, current_path, node_id)
                
                # Recursively scan nested structures
                if isinstance(value, (dict, list)):
                    self._scan_dict_for_secrets(value, current_path, node_id)
        
        elif isinstance(data, list):
            for i, item in enumerate(data):
                current_path = f"{path}[{i}]"
                self._scan_dict_for_secrets(item, current_path, node_id)
    
    def _scan_value_for_secrets(self, key: str, value: Any, path: str, node_id: Optional[str]):
        """Scan individual value for secret patterns"""
        if not isinstance(value, str):
            return
        
        # Skip if already a vault reference
        if self._is_vault_reference(value):
            return
        
        # Check against secret patterns
        for pattern in self.secret_patterns:
            matches = pattern.pattern.findall(value)
            if matches:
                self._report_secret_violation(
                    key, value, path, node_id, pattern, matches[0] if matches else ""
                )
    
    def _report_disallowed_field(self, field_name: str, path: str, value: Any, node_id: Optional[str]):
        """Report disallowed field usage"""
        self.secret_violations.append(SecretViolation(
            violation_code="SEC001",
            node_id=node_id,
            field_path=path,
            secret_type="disallowed_field",
            severity=SecretValidationSeverity.ERROR.value,
            message=f"Disallowed field '{field_name}' detected",
            detected_pattern=field_name,
            masked_value=self._mask_value(str(value)),
            expected="Use secure vault reference instead",
            actual=f"Direct field usage: {field_name}",
            autofix_available=True,
            suggested_vault_reference=self._generate_vault_reference(field_name, "disallowed_field"),
            remediation_steps=[
                f"Move {field_name} to secure vault",
                f"Replace with vault reference: vault://secrets/{field_name}",
                "Update access policies for vault secret"
            ]
        ))
    
    def _report_secret_violation(self, key: str, value: str, path: str, node_id: Optional[str], 
                               pattern: SecretPattern, matched_text: str):
        """Report secret pattern violation"""
        self.secret_violations.append(SecretViolation(
            violation_code=self._get_violation_code(pattern.secret_type),
            node_id=node_id,
            field_path=path,
            secret_type=pattern.secret_type,
            severity=SecretValidationSeverity.ERROR.value if pattern.confidence == "high" else SecretValidationSeverity.WARNING.value,
            message=f"{pattern.description} detected in field '{key}'",
            detected_pattern=pattern.name,
            masked_value=self._mask_value(value),
            expected="Secure vault reference (vault://...)",
            actual=f"Plaintext {pattern.secret_type}",
            autofix_available=True,
            suggested_vault_reference=self._generate_vault_reference(key, pattern.secret_type),
            remediation_steps=[
                f"Store {pattern.secret_type} in secure vault",
                f"Replace with vault reference: vault://{pattern.vault_prefix}/{key}",
                "Verify vault access policies",
                "Test vault reference resolution"
            ]
        ))
    
    def _is_vault_reference(self, value: str) -> bool:
        """Check if value is already a secure vault reference"""
        vault_prefixes = [
            'vault://',
            'kms://',
            'aws:secretsmanager:',
            'azure:keyvault:',
            'gcp:secretmanager:',
            '${vault.',
            '${aws:secretsmanager:',
            '${azure:keyvault:'
        ]
        
        return any(value.startswith(prefix) for prefix in vault_prefixes)
    
    def _mask_value(self, value: str) -> str:
        """Mask sensitive value for safe reporting"""
        if len(value) <= 8:
            return "*" * len(value)
        
        # Show first 2 and last 2 characters
        return value[:2] + "*" * (len(value) - 4) + value[-2:]
    
    def _get_violation_code(self, secret_type: str) -> str:
        """Get violation code for secret type"""
        code_mapping = {
            'api_key': 'SEC002',
            'password': 'SEC003',
            'token': 'SEC004',
            'certificate': 'SEC005',
            'connection_string': 'SEC006',
            'private_key': 'SEC007',
            'oauth_secret': 'SEC008',
            'database_url': 'SEC009'
        }
        return code_mapping.get(secret_type, 'SEC010')
    
    def _generate_vault_reference(self, key: str, secret_type: str) -> str:
        """Generate suggested vault reference"""
        vault_path_mapping = {
            'api_key': 'api-keys',
            'password': 'passwords',
            'token': 'tokens',
            'certificate': 'certificates',
            'connection_string': 'connections',
            'private_key': 'private-keys',
            'oauth_secret': 'oauth',
            'database_url': 'databases',
            'disallowed_field': 'secrets'
        }
        
        vault_path = vault_path_mapping.get(secret_type, 'secrets')
        return f"vault://{vault_path}/{key}"
    
    def _init_secret_patterns(self):
        """Initialize secret detection patterns"""
        self.secret_patterns = [
            # API Keys
            SecretPattern(
                name="generic_api_key",
                pattern=re.compile(r'["\']?(?:api[_-]?key|apikey)["\']?\s*[:=]\s*["\']?([a-zA-Z0-9_\-]{20,})["\']?', re.IGNORECASE),
                secret_type=SecretType.API_KEY.value,
                confidence="high",
                description="Generic API key pattern",
                vault_prefix="api-keys"
            ),
            
            # AWS Access Keys
            SecretPattern(
                name="aws_access_key",
                pattern=re.compile(r'AKIA[0-9A-Z]{16}'),
                secret_type=SecretType.API_KEY.value,
                confidence="high",
                description="AWS access key",
                vault_prefix="aws/access-keys"
            ),
            
            # AWS Secret Keys
            SecretPattern(
                name="aws_secret_key",
                pattern=re.compile(r'["\']?(?:aws[_-]?secret[_-]?access[_-]?key|aws[_-]?secret)["\']?\s*[:=]\s*["\']?([a-zA-Z0-9/+=]{40})["\']?', re.IGNORECASE),
                secret_type=SecretType.API_KEY.value,
                confidence="high",
                description="AWS secret access key",
                vault_prefix="aws/secret-keys"
            ),
            
            # Generic passwords
            SecretPattern(
                name="password_field",
                pattern=re.compile(r'["\']?(?:password|passwd|pwd)["\']?\s*[:=]\s*["\']?([^"\'\s]{8,})["\']?', re.IGNORECASE),
                secret_type=SecretType.PASSWORD.value,
                confidence="medium",
                description="Password field",
                vault_prefix="passwords"
            ),
            
            # JWT Tokens
            SecretPattern(
                name="jwt_token",
                pattern=re.compile(r'eyJ[a-zA-Z0-9_\-]*\.eyJ[a-zA-Z0-9_\-]*\.[a-zA-Z0-9_\-]*'),
                secret_type=SecretType.TOKEN.value,
                confidence="high",
                description="JWT token",
                vault_prefix="tokens/jwt"
            ),
            
            # Bearer tokens
            SecretPattern(
                name="bearer_token",
                pattern=re.compile(r'["\']?(?:bearer[_-]?token|bearer)["\']?\s*[:=]\s*["\']?([a-zA-Z0-9_\-]{20,})["\']?', re.IGNORECASE),
                secret_type=SecretType.TOKEN.value,
                confidence="high",
                description="Bearer token",
                vault_prefix="tokens/bearer"
            ),
            
            # OAuth secrets
            SecretPattern(
                name="oauth_secret",
                pattern=re.compile(r'["\']?(?:client[_-]?secret|oauth[_-]?secret)["\']?\s*[:=]\s*["\']?([a-zA-Z0-9_\-]{20,})["\']?', re.IGNORECASE),
                secret_type=SecretType.OAUTH_SECRET.value,
                confidence="high",
                description="OAuth client secret",
                vault_prefix="oauth/secrets"
            ),
            
            # Database connection strings
            SecretPattern(
                name="postgres_connection",
                pattern=re.compile(r'postgres(?:ql)?://[^:\s]+:[^@\s]+@[^/\s]+/[^\s]+', re.IGNORECASE),
                secret_type=SecretType.CONNECTION_STRING.value,
                confidence="high",
                description="PostgreSQL connection string with credentials",
                vault_prefix="databases/postgres"
            ),
            
            SecretPattern(
                name="mysql_connection",
                pattern=re.compile(r'mysql://[^:\s]+:[^@\s]+@[^/\s]+/[^\s]+', re.IGNORECASE),
                secret_type=SecretType.CONNECTION_STRING.value,
                confidence="high",
                description="MySQL connection string with credentials",
                vault_prefix="databases/mysql"
            ),
            
            # Private keys
            SecretPattern(
                name="rsa_private_key",
                pattern=re.compile(r'-----BEGIN (?:RSA )?PRIVATE KEY-----'),
                secret_type=SecretType.PRIVATE_KEY.value,
                confidence="high",
                description="RSA private key",
                vault_prefix="certificates/private-keys"
            ),
            
            SecretPattern(
                name="openssh_private_key",
                pattern=re.compile(r'-----BEGIN OPENSSH PRIVATE KEY-----'),
                secret_type=SecretType.PRIVATE_KEY.value,
                confidence="high",
                description="OpenSSH private key",
                vault_prefix="certificates/ssh-keys"
            ),
            
            # Certificates
            SecretPattern(
                name="certificate",
                pattern=re.compile(r'-----BEGIN CERTIFICATE-----'),
                secret_type=SecretType.CERTIFICATE.value,
                confidence="medium",
                description="X.509 certificate",
                vault_prefix="certificates/public"
            ),
            
            # Generic secrets (base64 encoded)
            SecretPattern(
                name="base64_secret",
                pattern=re.compile(r'["\']?(?:secret|key|token)["\']?\s*[:=]\s*["\']?([A-Za-z0-9+/]{40,}={0,2})["\']?', re.IGNORECASE),
                secret_type=SecretType.TOKEN.value,
                confidence="low",
                description="Potential base64-encoded secret",
                vault_prefix="secrets/encoded"
            ),
            
            # Slack webhooks
            SecretPattern(
                name="slack_webhook",
                pattern=re.compile(r'https://hooks\.slack\.com/services/[A-Z0-9]+/[A-Z0-9]+/[a-zA-Z0-9]+'),
                secret_type=SecretType.API_KEY.value,
                confidence="high",
                description="Slack webhook URL",
                vault_prefix="webhooks/slack"
            ),
            
            # GitHub tokens
            SecretPattern(
                name="github_token",
                pattern=re.compile(r'gh[pousr]_[A-Za-z0-9_]{36}'),
                secret_type=SecretType.TOKEN.value,
                confidence="high",
                description="GitHub personal access token",
                vault_prefix="tokens/github"
            )
        ]
    
    def _init_disallowed_fields(self):
        """Initialize list of disallowed field names"""
        self.disallowed_fields = {
            # Direct credential fields
            'password', 'passwd', 'pwd', 'secret', 'key', 'token',
            'api_key', 'apikey', 'access_key', 'secret_key',
            'client_secret', 'oauth_secret', 'auth_token',
            'bearer_token', 'jwt_token', 'refresh_token',
            
            # Database credentials
            'db_password', 'database_password', 'db_user', 'db_pass',
            'connection_string', 'database_url', 'db_url',
            
            # Cloud provider credentials
            'aws_access_key_id', 'aws_secret_access_key',
            'azure_client_secret', 'azure_tenant_id',
            'gcp_service_account_key', 'google_credentials',
            
            # Certificate/key fields
            'private_key', 'public_key', 'ssl_key', 'tls_key',
            'certificate', 'cert', 'ca_cert', 'ssl_cert',
            
            # Authentication fields
            'auth_header', 'authorization', 'x_api_key',
            'basic_auth', 'digest_auth', 'oauth_token',
            
            # Webhook/callback secrets
            'webhook_secret', 'callback_secret', 'signing_secret',
            'verification_token', 'webhook_url'
        }
    
    def generate_vault_migration_plan(self) -> Dict[str, Any]:
        """Generate plan for migrating secrets to vault"""
        migration_plan = {
            'total_secrets': len(self.secret_violations),
            'secrets_by_type': {},
            'vault_paths_needed': set(),
            'migration_steps': [],
            'estimated_effort_hours': 0
        }
        
        # Group by secret type
        for violation in self.secret_violations:
            secret_type = violation.secret_type
            if secret_type not in migration_plan['secrets_by_type']:
                migration_plan['secrets_by_type'][secret_type] = 0
            migration_plan['secrets_by_type'][secret_type] += 1
            
            # Track vault paths needed
            if violation.suggested_vault_reference:
                vault_path = violation.suggested_vault_reference.split('://')[1].split('/')[0]
                migration_plan['vault_paths_needed'].add(vault_path)
        
        # Generate migration steps
        migration_plan['migration_steps'] = [
            "1. Set up vault infrastructure and access policies",
            "2. Create vault paths for each secret type",
            "3. Migrate secrets to vault (by type):",
        ]
        
        for secret_type, count in migration_plan['secrets_by_type'].items():
            migration_plan['migration_steps'].append(f"   - {secret_type}: {count} secrets")
        
        migration_plan['migration_steps'].extend([
            "4. Update configurations to use vault references",
            "5. Test vault reference resolution",
            "6. Remove original secret values",
            "7. Verify no secrets remain in manifests"
        ])
        
        # Estimate effort (rough calculation)
        migration_plan['estimated_effort_hours'] = len(self.secret_violations) * 0.5 + len(migration_plan['vault_paths_needed']) * 2
        migration_plan['vault_paths_needed'] = list(migration_plan['vault_paths_needed'])
        
        return migration_plan
    
    def generate_autofix_script(self) -> str:
        """Generate script to automatically fix secret violations"""
        script_lines = [
            "#!/bin/bash",
            "# Auto-generated script to fix secret violations",
            "# WARNING: Review and test before running in production",
            "",
            "set -e",
            ""
        ]
        
        # Group violations by file/path
        violations_by_path = {}
        for violation in self.secret_violations:
            if violation.autofix_available:
                base_path = violation.field_path.split('.')[0]
                if base_path not in violations_by_path:
                    violations_by_path[base_path] = []
                violations_by_path[base_path].append(violation)
        
        # Generate fix commands for each path
        for path, violations in violations_by_path.items():
            script_lines.append(f"echo 'Fixing secrets in {path}...'")
            
            for violation in violations:
                if violation.suggested_vault_reference:
                    script_lines.append(f"# Fix {violation.secret_type} in {violation.field_path}")
                    script_lines.append(f"# Replace with: {violation.suggested_vault_reference}")
                    script_lines.append("")
        
        script_lines.extend([
            "echo 'Secret fixes completed.'",
            "echo 'Please verify vault references are working before deploying.'"
        ])
        
        return "\n".join(script_lines)
    
    def get_validation_summary(self) -> Dict[str, Any]:
        """Get summary of secrets validation results"""
        error_count = len([v for v in self.secret_violations if v.severity == 'error'])
        warning_count = len([v for v in self.secret_violations if v.severity == 'warning'])
        info_count = len([v for v in self.secret_violations if v.severity == 'info'])
        
        autofix_count = len([v for v in self.secret_violations if v.autofix_available])
        
        return {
            'total_violations': len(self.secret_violations),
            'errors': error_count,
            'warnings': warning_count,
            'info': info_count,
            'secrets_redaction_passed': error_count == 0,
            'autofix_available': autofix_count,
            'violations_by_type': self._group_violations_by_type(),
            'violations_by_node': self._group_violations_by_node(),
            'common_violations': self._get_common_violations(),
            'vault_migration_needed': error_count > 0,
            'estimated_migration_effort': f"{len(self.secret_violations) * 0.5:.1f} hours"
        }
    
    def _group_violations_by_type(self) -> Dict[str, int]:
        """Group violations by secret type"""
        type_counts = {}
        for violation in self.secret_violations:
            secret_type = violation.secret_type
            if secret_type not in type_counts:
                type_counts[secret_type] = 0
            type_counts[secret_type] += 1
        return type_counts
    
    def _group_violations_by_node(self) -> Dict[str, int]:
        """Group violations by node"""
        node_counts = {}
        for violation in self.secret_violations:
            node_id = violation.node_id or 'global'
            if node_id not in node_counts:
                node_counts[node_id] = 0
            node_counts[node_id] += 1
        return node_counts
    
    def _get_common_violations(self) -> List[str]:
        """Get most common secret violations"""
        violation_descriptions = {
            'SEC001': 'Disallowed field usage',
            'SEC002': 'API key exposure',
            'SEC003': 'Password exposure',
            'SEC004': 'Token exposure',
            'SEC005': 'Certificate exposure',
            'SEC006': 'Connection string exposure',
            'SEC007': 'Private key exposure',
            'SEC008': 'OAuth secret exposure',
            'SEC009': 'Database URL exposure',
            'SEC010': 'Generic secret exposure'
        }
        
        violation_counts = {}
        for violation in self.secret_violations:
            code = violation.violation_code
            if code not in violation_counts:
                violation_counts[code] = 0
            violation_counts[code] += 1
        
        sorted_violations = sorted(violation_counts.items(), key=lambda x: x[1], reverse=True)
        
        return [f"{violation_descriptions.get(code, code)}: {count} occurrences" 
                for code, count in sorted_violations[:5]]

# Test helper functions
def create_test_nodes_with_secrets():
    """Create test nodes with various secret exposure issues"""
    from .ir import IRNode, IRNodeType
    
    nodes = [
        # Node with API key exposure
        IRNode(
            id="api_key_exposure",
            type=IRNodeType.ML_PREDICT,
            parameters={
                'model_id': 'test_model',
                'api_key': 'sk-1234567890abcdef1234567890abcdef',  # Exposed API key
                'endpoint': 'https://api.example.com/v1',
                'headers': {
                    'Authorization': 'Bearer eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...'  # JWT token
                }
            }
        ),
        
        # Node with database credentials
        IRNode(
            id="db_credentials_exposure",
            type=IRNodeType.QUERY,
            parameters={
                'connection_string': 'postgresql://user:password123@db.example.com:5432/prod',  # DB credentials
                'password': 'secretpassword123',  # Direct password
                'db_config': {
                    'host': 'db.example.com',
                    'user': 'dbuser',
                    'password': 'dbpass456'  # Nested password
                }
            }
        ),
        
        # Node with proper vault references (should not trigger violations)
        IRNode(
            id="proper_vault_usage",
            type=IRNodeType.ML_SCORE,
            parameters={
                'model_id': 'secure_model',
                'api_key': 'vault://api-keys/ml-service-key',
                'database_url': 'vault://databases/postgres/prod-connection',
                'oauth_secret': 'vault://oauth/secrets/client-secret'
            }
        ),
        
        # Node with certificate exposure
        IRNode(
            id="certificate_exposure",
            type=IRNodeType.ACTION,
            parameters={
                'ssl_config': {
                    'private_key': '-----BEGIN RSA PRIVATE KEY-----\nMIIEpAIBAAKCAQEA...',  # Private key
                    'certificate': '-----BEGIN CERTIFICATE-----\nMIIDXTCCAkWgAwIBAgIJAK...'  # Certificate
                }
            },
            metadata={
                'webhook_secret': 'whsec_1234567890abcdef1234567890abcdef',  # Webhook secret
                'github_token': 'ghp_1234567890abcdef1234567890abcdef123456'  # GitHub token
            }
        )
    ]
    
    return nodes

def create_test_manifest_with_secrets():
    """Create test manifest with secret exposures"""
    return {
        'workflow_id': 'test_secrets_workflow',
        'version': '1.0.0',
        'created_at': '2024-01-15T10:00:00Z',
        'deployment_config': {
            'aws_access_key_id': 'AKIAIOSFODNN7EXAMPLE',  # AWS access key
            'aws_secret_access_key': 'wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY',  # AWS secret
            'slack_webhook': 'https://hooks.slack.com/services/T00000000/B00000000/XXXXXXXXXXXXXXXXXXXXXXXX'  # Slack webhook
        },
        'database_config': {
            'url': 'mysql://root:password@localhost:3306/mydb',  # MySQL with password
            'backup_url': 'vault://databases/mysql/backup-connection'  # Proper vault reference
        }
    }

def run_secrets_redaction_tests():
    """Run secrets redaction validation tests"""
    validator = SecretsRedactionValidator()
    
    # Create test graph and manifest
    test_nodes = create_test_nodes_with_secrets()
    test_manifest = create_test_manifest_with_secrets()
    
    from .ir import IRGraph
    
    test_graph = IRGraph(
        workflow_id="test_secrets_validation",
        name="Test Secrets Validation Workflow",
        version="1.0.0",
        automation_type="rbia",
        nodes=test_nodes
    )
    
    # Run validation
    violations = validator.validate_secrets_redaction(test_graph, test_manifest)
    
    print(f"Secrets redaction validation completed. Found {len(violations)} violations:")
    for violation in violations:
        print(f"  {violation.violation_code} [{violation.severity}] {violation.field_path}: {violation.message}")
        print(f"    Detected: {violation.detected_pattern}")
        print(f"    Masked value: {violation.masked_value}")
        if violation.autofix_available:
            print(f"    Suggested fix: {violation.suggested_vault_reference}")
    
    # Print summary
    summary = validator.get_validation_summary()
    print(f"\nValidation Summary:")
    print(f"  Secrets redaction passed: {summary['secrets_redaction_passed']}")
    print(f"  Errors: {summary['errors']}")
    print(f"  Warnings: {summary['warnings']}")
    print(f"  Autofix available: {summary['autofix_available']}")
    print(f"  Vault migration needed: {summary['vault_migration_needed']}")
    
    # Generate migration plan
    migration_plan = validator.generate_vault_migration_plan()
    print(f"\nVault Migration Plan:")
    print(f"  Total secrets to migrate: {migration_plan['total_secrets']}")
    print(f"  Vault paths needed: {', '.join(migration_plan['vault_paths_needed'])}")
    print(f"  Estimated effort: {migration_plan['estimated_effort_hours']} hours")
    
    # Generate autofix script
    autofix_script = validator.generate_autofix_script()
    print(f"\nAutofix script generated ({len(autofix_script.split())} lines)")
    
    return violations, summary

if __name__ == "__main__":
    run_secrets_redaction_tests()
