"""
Secure Import - Task 6.2.65
============================

Supply chain security with checksum libs and signed policies
- Validates checksums of imported libraries
- Verifies signatures of policy packs
- CI integration for security verification
"""

from typing import Dict, List, Any, Optional
import logging
import hashlib
import json
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)

class ImportType(Enum):
    LIBRARY = "library"
    POLICY_PACK = "policy_pack"
    TEMPLATE = "template"
    SCHEMA = "schema"

class SecurityLevel(Enum):
    NONE = "none"
    CHECKSUM = "checksum"
    SIGNATURE = "signature"
    FULL = "full"  # Both checksum and signature

@dataclass
class ImportManifest:
    import_id: str
    import_type: ImportType
    name: str
    version: str
    source_url: str
    checksum_sha256: str
    signature: Optional[str] = None
    public_key_id: Optional[str] = None
    security_level: SecurityLevel = SecurityLevel.CHECKSUM

@dataclass
class SecurityViolation:
    violation_type: str
    import_id: str
    description: str
    severity: str  # critical, high, medium, low

class SecureImportService:
    """Service for secure import validation"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.trusted_sources = self._initialize_trusted_sources()
        self.public_keys = self._initialize_public_keys()
    
    def _initialize_trusted_sources(self) -> Dict[str, Dict[str, Any]]:
        """Initialize trusted source registry"""
        return {
            'official_rbia_repo': {
                'base_url': 'https://repo.rbia.io/',
                'required_security_level': SecurityLevel.SIGNATURE,
                'public_key_id': 'rbia_official_2024'
            },
            'community_templates': {
                'base_url': 'https://templates.rbia.community/',
                'required_security_level': SecurityLevel.CHECKSUM,
                'public_key_id': None
            },
            'internal_registry': {
                'base_url': 'https://internal.company.com/rbia/',
                'required_security_level': SecurityLevel.FULL,
                'public_key_id': 'company_internal_2024'
            }
        }
    
    def _initialize_public_keys(self) -> Dict[str, str]:
        """Initialize public keys for signature verification"""
        return {
            'rbia_official_2024': '''-----BEGIN PUBLIC KEY-----
MIIBIjANBgkqhkiG9w0BAQEFAAOCAQ8AMIIBCgKCAQEA...
-----END PUBLIC KEY-----''',
            'company_internal_2024': '''-----BEGIN PUBLIC KEY-----
MIIBIjANBgkqhkiG9w0BAQEFAAOCAQ8AMIIBCgKCAQEA...
-----END PUBLIC KEY-----'''
        }
    
    def validate_import(self, import_manifest: ImportManifest, 
                       content: bytes) -> List[SecurityViolation]:
        """Validate an import against security requirements"""
        violations = []
        
        # Validate checksum
        checksum_violation = self._validate_checksum(import_manifest, content)
        if checksum_violation:
            violations.append(checksum_violation)
        
        # Validate signature if required
        if import_manifest.security_level in [SecurityLevel.SIGNATURE, SecurityLevel.FULL]:
            signature_violation = self._validate_signature(import_manifest, content)
            if signature_violation:
                violations.append(signature_violation)
        
        # Validate source
        source_violation = self._validate_source(import_manifest)
        if source_violation:
            violations.append(source_violation)
        
        return violations
    
    def _validate_checksum(self, import_manifest: ImportManifest, 
                          content: bytes) -> Optional[SecurityViolation]:
        """Validate content checksum"""
        calculated_checksum = hashlib.sha256(content).hexdigest()
        
        if calculated_checksum != import_manifest.checksum_sha256:
            return SecurityViolation(
                violation_type="checksum_mismatch",
                import_id=import_manifest.import_id,
                description=f"Checksum mismatch for {import_manifest.name}. Expected: {import_manifest.checksum_sha256}, Got: {calculated_checksum}",
                severity="critical"
            )
        
        return None
    
    def _validate_signature(self, import_manifest: ImportManifest, 
                           content: bytes) -> Optional[SecurityViolation]:
        """Validate content signature"""
        if not import_manifest.signature or not import_manifest.public_key_id:
            return SecurityViolation(
                violation_type="missing_signature",
                import_id=import_manifest.import_id,
                description=f"Signature required but missing for {import_manifest.name}",
                severity="critical"
            )
        
        public_key = self.public_keys.get(import_manifest.public_key_id)
        if not public_key:
            return SecurityViolation(
                violation_type="unknown_public_key",
                import_id=import_manifest.import_id,
                description=f"Unknown public key ID: {import_manifest.public_key_id}",
                severity="high"
            )
        
        # Simulate signature verification (would use actual crypto library)
        signature_valid = self._verify_signature_mock(content, import_manifest.signature, public_key)
        
        if not signature_valid:
            return SecurityViolation(
                violation_type="invalid_signature",
                import_id=import_manifest.import_id,
                description=f"Invalid signature for {import_manifest.name}",
                severity="critical"
            )
        
        return None
    
    def _validate_source(self, import_manifest: ImportManifest) -> Optional[SecurityViolation]:
        """Validate import source"""
        # Check if source is from trusted registry
        trusted_source = None
        for source_name, source_config in self.trusted_sources.items():
            if import_manifest.source_url.startswith(source_config['base_url']):
                trusted_source = source_config
                break
        
        if not trusted_source:
            return SecurityViolation(
                violation_type="untrusted_source",
                import_id=import_manifest.import_id,
                description=f"Import from untrusted source: {import_manifest.source_url}",
                severity="medium"
            )
        
        # Check if security level meets requirements
        required_level = trusted_source['required_security_level']
        if self._security_level_insufficient(import_manifest.security_level, required_level):
            return SecurityViolation(
                violation_type="insufficient_security_level",
                import_id=import_manifest.import_id,
                description=f"Security level {import_manifest.security_level.value} insufficient, requires {required_level.value}",
                severity="high"
            )
        
        return None
    
    def _verify_signature_mock(self, content: bytes, signature: str, public_key: str) -> bool:
        """Mock signature verification (would use actual crypto library)"""
        # In real implementation, would use cryptography library
        # For now, simulate based on content hash
        content_hash = hashlib.sha256(content).hexdigest()
        expected_signature = hashlib.sha256(f"{content_hash}{public_key}".encode()).hexdigest()[:32]
        return signature == expected_signature
    
    def _security_level_insufficient(self, current: SecurityLevel, required: SecurityLevel) -> bool:
        """Check if current security level is insufficient"""
        level_hierarchy = {
            SecurityLevel.NONE: 0,
            SecurityLevel.CHECKSUM: 1,
            SecurityLevel.SIGNATURE: 2,
            SecurityLevel.FULL: 3
        }
        
        return level_hierarchy[current] < level_hierarchy[required]
    
    def create_import_manifest(self, import_type: ImportType, name: str, version: str,
                              source_url: str, content: bytes,
                              signature: Optional[str] = None,
                              public_key_id: Optional[str] = None) -> ImportManifest:
        """Create import manifest for a library or policy pack"""
        checksum = hashlib.sha256(content).hexdigest()
        import_id = f"{import_type.value}_{name}_{version}_{checksum[:8]}"
        
        security_level = SecurityLevel.CHECKSUM
        if signature and public_key_id:
            security_level = SecurityLevel.FULL
        elif signature:
            security_level = SecurityLevel.SIGNATURE
        
        return ImportManifest(
            import_id=import_id,
            import_type=import_type,
            name=name,
            version=version,
            source_url=source_url,
            checksum_sha256=checksum,
            signature=signature,
            public_key_id=public_key_id,
            security_level=security_level
        )
    
    def validate_policy_pack(self, policy_pack_data: Dict[str, Any], 
                           manifest: ImportManifest) -> List[SecurityViolation]:
        """Validate policy pack with additional checks"""
        violations = []
        
        # Basic import validation
        policy_content = json.dumps(policy_pack_data, sort_keys=True).encode()
        basic_violations = self.validate_import(manifest, policy_content)
        violations.extend(basic_violations)
        
        # Policy-specific validations
        if 'version' not in policy_pack_data:
            violations.append(SecurityViolation(
                violation_type="missing_policy_version",
                import_id=manifest.import_id,
                description="Policy pack missing version information",
                severity="medium"
            ))
        
        if 'author' not in policy_pack_data:
            violations.append(SecurityViolation(
                violation_type="missing_policy_author",
                import_id=manifest.import_id,
                description="Policy pack missing author information",
                severity="low"
            ))
        
        # Check for dangerous policy configurations
        dangerous_patterns = ['allow_all', 'disable_security', 'bypass_approval']
        policy_text = json.dumps(policy_pack_data).lower()
        
        for pattern in dangerous_patterns:
            if pattern in policy_text:
                violations.append(SecurityViolation(
                    violation_type="dangerous_policy_configuration",
                    import_id=manifest.import_id,
                    description=f"Policy contains potentially dangerous configuration: {pattern}",
                    severity="high"
                ))
        
        return violations
    
    def generate_ci_report(self, import_manifests: List[ImportManifest],
                          violations: List[SecurityViolation]) -> Dict[str, Any]:
        """Generate CI report for security validation"""
        critical_violations = [v for v in violations if v.severity == "critical"]
        high_violations = [v for v in violations if v.severity == "high"]
        
        should_block_ci = len(critical_violations) > 0
        
        return {
            'validation_status': 'failed' if should_block_ci else 'passed',
            'should_block_ci': should_block_ci,
            'total_imports': len(import_manifests),
            'total_violations': len(violations),
            'critical_violations': len(critical_violations),
            'high_violations': len(high_violations),
            'violations_by_type': self._group_violations_by_type(violations),
            'recommendations': self._generate_security_recommendations(violations),
            'imports_summary': [
                {
                    'import_id': manifest.import_id,
                    'name': manifest.name,
                    'type': manifest.import_type.value,
                    'security_level': manifest.security_level.value,
                    'violations': len([v for v in violations if v.import_id == manifest.import_id])
                }
                for manifest in import_manifests
            ]
        }
    
    def _group_violations_by_type(self, violations: List[SecurityViolation]) -> Dict[str, int]:
        """Group violations by type"""
        violation_counts = {}
        for violation in violations:
            violation_type = violation.violation_type
            violation_counts[violation_type] = violation_counts.get(violation_type, 0) + 1
        return violation_counts
    
    def _generate_security_recommendations(self, violations: List[SecurityViolation]) -> List[str]:
        """Generate security recommendations based on violations"""
        recommendations = []
        
        violation_types = {v.violation_type for v in violations}
        
        if 'checksum_mismatch' in violation_types:
            recommendations.append("Verify import sources and update checksums")
        
        if 'invalid_signature' in violation_types:
            recommendations.append("Ensure imports are properly signed with valid keys")
        
        if 'untrusted_source' in violation_types:
            recommendations.append("Use imports only from trusted registries")
        
        if 'dangerous_policy_configuration' in violation_types:
            recommendations.append("Review policy configurations for security implications")
        
        return recommendations
    
    def get_trusted_sources(self) -> Dict[str, Dict[str, Any]]:
        """Get list of trusted sources"""
        return self.trusted_sources.copy()
    
    def add_trusted_source(self, name: str, base_url: str, 
                          security_level: SecurityLevel,
                          public_key_id: Optional[str] = None):
        """Add a new trusted source"""
        self.trusted_sources[name] = {
            'base_url': base_url,
            'required_security_level': security_level,
            'public_key_id': public_key_id
        }
        
        self.logger.info(f"Added trusted source: {name} ({base_url})")

