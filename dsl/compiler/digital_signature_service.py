"""
Digital Signature Service - Task 6.2.28
========================================

Digital signature of manifest (backend signing logic only)
- Cryptographically sign manifest for tamper detection and provenance verification
- Backend signing and verification logic (no KMS/HSM integration - that's infrastructure)
- Environment-specific signing metadata (key id, timestamp, environment)
- Signature verification API for manifest authenticity
- Policy validation for signature requirements

Dependencies: Task 6.2.27 (Plan Hash Service)
Outputs: Cryptographic signatures → enables tamper detection and provenance verification
"""

from typing import Dict, List, Any, Optional, Tuple, Union
import hashlib
import json
import logging
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum
import base64
import secrets
import hmac
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import rsa, padding
from cryptography.hazmat.primitives.serialization import load_pem_private_key, load_pem_public_key
import time

logger = logging.getLogger(__name__)

class SigningEnvironment(Enum):
    """Environment types for signing"""
    DEVELOPMENT = "development"
    STAGING = "staging"
    PRODUCTION = "production"
    SANDBOX = "sandbox"

class SignatureAlgorithm(Enum):
    """Supported signature algorithms"""
    RSA_PSS_SHA256 = "rsa_pss_sha256"
    RSA_PKCS1_SHA256 = "rsa_pkcs1_sha256"
    HMAC_SHA256 = "hmac_sha256"  # For development/testing

@dataclass
class SigningKey:
    """Signing key metadata"""
    key_id: str
    algorithm: SignatureAlgorithm
    environment: SigningEnvironment
    created_at: datetime
    expires_at: Optional[datetime] = None
    is_active: bool = True
    key_fingerprint: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class ManifestSignature:
    """Digital signature for a manifest"""
    signature_id: str
    manifest_hash: str
    signature_bytes: str  # Base64 encoded
    algorithm: SignatureAlgorithm
    key_id: str
    environment: SigningEnvironment
    signed_at: datetime
    signer_metadata: Dict[str, Any] = field(default_factory=dict)
    verification_status: str = "unverified"

@dataclass
class SignatureVerificationResult:
    """Result of signature verification"""
    is_valid: bool
    signature_id: str
    verified_at: datetime
    key_id: str
    algorithm: SignatureAlgorithm
    error_message: Optional[str] = None
    trust_level: str = "unknown"  # high, medium, low based on key/env

@dataclass
class SigningPolicy:
    """Policy for signature requirements"""
    environment: SigningEnvironment
    required_algorithm: SignatureAlgorithm
    require_signature: bool = True
    max_signature_age_hours: int = 24 * 30  # 30 days
    trusted_key_ids: List[str] = field(default_factory=list)
    metadata_requirements: List[str] = field(default_factory=list)

# Task 6.2.28: Digital Signature Service (Backend Only)
class DigitalSignatureService:
    """Service for signing and verifying plan manifests (backend implementation)"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
        # Backend storage for keys and signatures
        self.signing_keys: Dict[str, SigningKey] = {}
        self.signatures: Dict[str, ManifestSignature] = {}  # signature_id -> signature
        self.manifest_signatures: Dict[str, List[str]] = {}  # manifest_hash -> signature_ids
        
        # Signing policies by environment
        self.signing_policies: Dict[SigningEnvironment, SigningPolicy] = {
            SigningEnvironment.DEVELOPMENT: SigningPolicy(
                environment=SigningEnvironment.DEVELOPMENT,
                required_algorithm=SignatureAlgorithm.HMAC_SHA256,
                require_signature=False
            ),
            SigningEnvironment.STAGING: SigningPolicy(
                environment=SigningEnvironment.STAGING,
                required_algorithm=SignatureAlgorithm.RSA_PKCS1_SHA256,
                require_signature=True
            ),
            SigningEnvironment.PRODUCTION: SigningPolicy(
                environment=SigningEnvironment.PRODUCTION,
                required_algorithm=SignatureAlgorithm.RSA_PSS_SHA256,
                require_signature=True,
                max_signature_age_hours=24 * 7  # 7 days for prod
            )
        }
        
        # In-memory key storage (simulating key management)
        self.private_keys: Dict[str, bytes] = {}  # key_id -> private key bytes
        self.public_keys: Dict[str, bytes] = {}   # key_id -> public key bytes
        self.hmac_keys: Dict[str, bytes] = {}     # key_id -> hmac key
        
        # Statistics
        self.signature_stats = {
            'total_signatures_created': 0,
            'total_verifications': 0,
            'verification_failures': 0,
            'signatures_by_environment': {env.value: 0 for env in SigningEnvironment}
        }
    
    def generate_signing_key(self, key_id: str, 
                           algorithm: SignatureAlgorithm,
                           environment: SigningEnvironment,
                           expires_in_days: Optional[int] = None) -> SigningKey:
        """
        Generate a new signing key (backend implementation)
        
        Args:
            key_id: Unique identifier for the key
            algorithm: Signature algorithm
            environment: Environment this key is for
            expires_in_days: Key expiration in days
            
        Returns:
            SigningKey metadata
        """
        expires_at = None
        if expires_in_days:
            expires_at = datetime.utcnow() + timedelta(days=expires_in_days)
        
        # Generate key based on algorithm
        if algorithm in [SignatureAlgorithm.RSA_PSS_SHA256, SignatureAlgorithm.RSA_PKCS1_SHA256]:
            # Generate RSA key pair
            private_key = rsa.generate_private_key(
                public_exponent=65537,
                key_size=2048
            )
            public_key = private_key.public_key()
            
            # Serialize keys
            private_pem = private_key.private_bytes(
                encoding=serialization.Encoding.PEM,
                format=serialization.PrivateFormat.PKCS8,
                encryption_algorithm=serialization.NoEncryption()
            )
            public_pem = public_key.public_bytes(
                encoding=serialization.Encoding.PEM,
                format=serialization.PublicFormat.SubjectPublicKeyInfo
            )
            
            self.private_keys[key_id] = private_pem
            self.public_keys[key_id] = public_pem
            
            # Create fingerprint
            key_fingerprint = hashlib.sha256(public_pem).hexdigest()[:16]
            
        elif algorithm == SignatureAlgorithm.HMAC_SHA256:
            # Generate HMAC key
            hmac_key = secrets.token_bytes(32)  # 256-bit key
            self.hmac_keys[key_id] = hmac_key
            key_fingerprint = hashlib.sha256(hmac_key).hexdigest()[:16]
        
        else:
            raise ValueError(f"Unsupported algorithm: {algorithm}")
        
        # Create signing key metadata
        signing_key = SigningKey(
            key_id=key_id,
            algorithm=algorithm,
            environment=environment,
            created_at=datetime.utcnow(),
            expires_at=expires_at,
            key_fingerprint=key_fingerprint,
            metadata={
                'generated_by': 'digital_signature_service',
                'key_size': 2048 if 'RSA' in algorithm.value else 256
            }
        )
        
        self.signing_keys[key_id] = signing_key
        
        self.logger.info(f"✅ Generated {algorithm.value} signing key: {key_id} for {environment.value}")
        
        return signing_key
    
    def sign_manifest(self, manifest_hash: str,
                     key_id: str,
                     environment: SigningEnvironment,
                     signer_metadata: Optional[Dict[str, Any]] = None) -> ManifestSignature:
        """
        Sign a manifest hash
        
        Args:
            manifest_hash: Hash of the canonical manifest
            key_id: Signing key identifier
            environment: Signing environment
            signer_metadata: Additional metadata about the signer
            
        Returns:
            ManifestSignature with signature details
        """
        # Validate key exists and is active
        if key_id not in self.signing_keys:
            raise ValueError(f"Signing key not found: {key_id}")
        
        signing_key = self.signing_keys[key_id]
        if not signing_key.is_active:
            raise ValueError(f"Signing key is inactive: {key_id}")
        
        if signing_key.expires_at and signing_key.expires_at < datetime.utcnow():
            raise ValueError(f"Signing key has expired: {key_id}")
        
        # Check signing policy
        policy = self.signing_policies.get(environment)
        if policy and policy.require_signature and signing_key.algorithm != policy.required_algorithm:
            raise ValueError(f"Algorithm {signing_key.algorithm.value} not allowed for {environment.value}")
        
        # Create signature data to sign
        signature_data = {
            'manifest_hash': manifest_hash,
            'key_id': key_id,
            'environment': environment.value,
            'signed_at': datetime.utcnow().isoformat()
        }
        
        # Add signer metadata if provided
        if signer_metadata:
            signature_data['signer_metadata'] = signer_metadata
        
        # Convert to canonical JSON for signing
        canonical_data = json.dumps(signature_data, sort_keys=True, separators=(',', ':')).encode('utf-8')
        
        # Sign based on algorithm
        if signing_key.algorithm == SignatureAlgorithm.RSA_PSS_SHA256:
            signature_bytes = self._sign_rsa_pss(canonical_data, key_id)
        elif signing_key.algorithm == SignatureAlgorithm.RSA_PKCS1_SHA256:
            signature_bytes = self._sign_rsa_pkcs1(canonical_data, key_id)
        elif signing_key.algorithm == SignatureAlgorithm.HMAC_SHA256:
            signature_bytes = self._sign_hmac(canonical_data, key_id)
        else:
            raise ValueError(f"Unsupported signature algorithm: {signing_key.algorithm}")
        
        # Create signature object
        signature_id = f"sig_{secrets.token_hex(8)}"
        signature = ManifestSignature(
            signature_id=signature_id,
            manifest_hash=manifest_hash,
            signature_bytes=base64.b64encode(signature_bytes).decode('utf-8'),
            algorithm=signing_key.algorithm,
            key_id=key_id,
            environment=environment,
            signed_at=datetime.utcnow(),
            signer_metadata=signer_metadata or {}
        )
        
        # Store signature
        self.signatures[signature_id] = signature
        
        # Track signatures by manifest
        if manifest_hash not in self.manifest_signatures:
            self.manifest_signatures[manifest_hash] = []
        self.manifest_signatures[manifest_hash].append(signature_id)
        
        # Update statistics
        self.signature_stats['total_signatures_created'] += 1
        self.signature_stats['signatures_by_environment'][environment.value] += 1
        
        self.logger.info(f"✅ Signed manifest {manifest_hash[:16]}... with key {key_id}")
        
        return signature
    
    def verify_signature(self, signature_id: str, 
                        manifest_hash: str) -> SignatureVerificationResult:
        """
        Verify a manifest signature
        
        Args:
            signature_id: Signature identifier
            manifest_hash: Expected manifest hash
            
        Returns:
            SignatureVerificationResult with verification status
        """
        try:
            # Get signature
            if signature_id not in self.signatures:
                return SignatureVerificationResult(
                    is_valid=False,
                    signature_id=signature_id,
                    verified_at=datetime.utcnow(),
                    key_id="unknown",
                    algorithm=SignatureAlgorithm.RSA_PSS_SHA256,
                    error_message="Signature not found"
                )
            
            signature = self.signatures[signature_id]
            
            # Verify manifest hash matches
            if signature.manifest_hash != manifest_hash:
                return SignatureVerificationResult(
                    is_valid=False,
                    signature_id=signature_id,
                    verified_at=datetime.utcnow(),
                    key_id=signature.key_id,
                    algorithm=signature.algorithm,
                    error_message=f"Manifest hash mismatch: expected {manifest_hash[:16]}..., got {signature.manifest_hash[:16]}..."
                )
            
            # Get signing key
            if signature.key_id not in self.signing_keys:
                return SignatureVerificationResult(
                    is_valid=False,
                    signature_id=signature_id,
                    verified_at=datetime.utcnow(),
                    key_id=signature.key_id,
                    algorithm=signature.algorithm,
                    error_message="Signing key not found"
                )
            
            signing_key = self.signing_keys[signature.key_id]
            
            # Check key expiration
            if signing_key.expires_at and signing_key.expires_at < datetime.utcnow():
                return SignatureVerificationResult(
                    is_valid=False,
                    signature_id=signature_id,
                    verified_at=datetime.utcnow(),
                    key_id=signature.key_id,
                    algorithm=signature.algorithm,
                    error_message="Signing key has expired"
                )
            
            # Recreate signature data
            signature_data = {
                'manifest_hash': signature.manifest_hash,
                'key_id': signature.key_id,
                'environment': signature.environment.value,
                'signed_at': signature.signed_at.isoformat()
            }
            
            if signature.signer_metadata:
                signature_data['signer_metadata'] = signature.signer_metadata
            
            canonical_data = json.dumps(signature_data, sort_keys=True, separators=(',', ':')).encode('utf-8')
            signature_bytes = base64.b64decode(signature.signature_bytes)
            
            # Verify signature based on algorithm
            is_valid = False
            if signature.algorithm == SignatureAlgorithm.RSA_PSS_SHA256:
                is_valid = self._verify_rsa_pss(canonical_data, signature_bytes, signature.key_id)
            elif signature.algorithm == SignatureAlgorithm.RSA_PKCS1_SHA256:
                is_valid = self._verify_rsa_pkcs1(canonical_data, signature_bytes, signature.key_id)
            elif signature.algorithm == SignatureAlgorithm.HMAC_SHA256:
                is_valid = self._verify_hmac(canonical_data, signature_bytes, signature.key_id)
            
            # Determine trust level
            trust_level = self._determine_trust_level(signature.environment, signing_key.algorithm)
            
            result = SignatureVerificationResult(
                is_valid=is_valid,
                signature_id=signature_id,
                verified_at=datetime.utcnow(),
                key_id=signature.key_id,
                algorithm=signature.algorithm,
                trust_level=trust_level
            )
            
            if not is_valid:
                result.error_message = "Signature verification failed"
                self.signature_stats['verification_failures'] += 1
            
            self.signature_stats['total_verifications'] += 1
            
            return result
            
        except Exception as e:
            self.signature_stats['verification_failures'] += 1
            return SignatureVerificationResult(
                is_valid=False,
                signature_id=signature_id,
                verified_at=datetime.utcnow(),
                key_id="unknown",
                algorithm=SignatureAlgorithm.RSA_PSS_SHA256,
                error_message=f"Verification error: {str(e)}"
            )
    
    def get_manifest_signatures(self, manifest_hash: str) -> List[ManifestSignature]:
        """Get all signatures for a manifest"""
        signature_ids = self.manifest_signatures.get(manifest_hash, [])
        return [self.signatures[sig_id] for sig_id in signature_ids if sig_id in self.signatures]
    
    def validate_signature_policy(self, manifest_hash: str, 
                                 environment: SigningEnvironment) -> Dict[str, Any]:
        """
        Validate that manifest meets signature policy requirements
        
        Args:
            manifest_hash: Manifest to check
            environment: Target environment
            
        Returns:
            Dictionary with policy validation results
        """
        policy = self.signing_policies.get(environment)
        if not policy:
            return {
                'is_valid': True,
                'reason': 'No policy defined for environment',
                'environment': environment.value
            }
        
        signatures = self.get_manifest_signatures(manifest_hash)
        
        # Check if signature is required
        if policy.require_signature and not signatures:
            return {
                'is_valid': False,
                'reason': f'Signature required for {environment.value} but none found',
                'environment': environment.value,
                'required_algorithm': policy.required_algorithm.value
            }
        
        if not policy.require_signature:
            return {
                'is_valid': True,
                'reason': 'Signature not required for environment',
                'environment': environment.value
            }
        
        # Check signature algorithm and age
        valid_signatures = []
        for signature in signatures:
            # Check algorithm
            if signature.algorithm != policy.required_algorithm:
                continue
            
            # Check age
            age_hours = (datetime.utcnow() - signature.signed_at).total_seconds() / 3600
            if age_hours > policy.max_signature_age_hours:
                continue
            
            # Check trusted keys
            if policy.trusted_key_ids and signature.key_id not in policy.trusted_key_ids:
                continue
            
            valid_signatures.append(signature)
        
        if not valid_signatures:
            return {
                'is_valid': False,
                'reason': 'No valid signatures found meeting policy requirements',
                'environment': environment.value,
                'required_algorithm': policy.required_algorithm.value,
                'max_age_hours': policy.max_signature_age_hours,
                'signatures_found': len(signatures)
            }
        
        return {
            'is_valid': True,
            'reason': 'Policy requirements met',
            'environment': environment.value,
            'valid_signatures': len(valid_signatures),
            'total_signatures': len(signatures)
        }
    
    def get_signature_statistics(self) -> Dict[str, Any]:
        """Get signature service statistics"""
        active_keys = len([k for k in self.signing_keys.values() if k.is_active])
        expired_keys = len([k for k in self.signing_keys.values() 
                           if k.expires_at and k.expires_at < datetime.utcnow()])
        
        return {
            **self.signature_stats,
            'total_signing_keys': len(self.signing_keys),
            'active_keys': active_keys,
            'expired_keys': expired_keys,
            'total_stored_signatures': len(self.signatures),
            'success_rate': (
                (self.signature_stats['total_verifications'] - self.signature_stats['verification_failures'])
                / max(1, self.signature_stats['total_verifications'])
            ) * 100
        }
    
    def _sign_rsa_pss(self, data: bytes, key_id: str) -> bytes:
        """Sign data using RSA-PSS"""
        private_key_pem = self.private_keys[key_id]
        private_key = load_pem_private_key(private_key_pem, password=None)
        
        signature = private_key.sign(
            data,
            padding.PSS(
                mgf=padding.MGF1(hashes.SHA256()),
                salt_length=padding.PSS.MAX_LENGTH
            ),
            hashes.SHA256()
        )
        return signature
    
    def _verify_rsa_pss(self, data: bytes, signature: bytes, key_id: str) -> bool:
        """Verify RSA-PSS signature"""
        try:
            public_key_pem = self.public_keys[key_id]
            public_key = load_pem_public_key(public_key_pem)
            
            public_key.verify(
                signature,
                data,
                padding.PSS(
                    mgf=padding.MGF1(hashes.SHA256()),
                    salt_length=padding.PSS.MAX_LENGTH
                ),
                hashes.SHA256()
            )
            return True
        except Exception:
            return False
    
    def _sign_rsa_pkcs1(self, data: bytes, key_id: str) -> bytes:
        """Sign data using RSA PKCS#1 v1.5"""
        private_key_pem = self.private_keys[key_id]
        private_key = load_pem_private_key(private_key_pem, password=None)
        
        signature = private_key.sign(
            data,
            padding.PKCS1v15(),
            hashes.SHA256()
        )
        return signature
    
    def _verify_rsa_pkcs1(self, data: bytes, signature: bytes, key_id: str) -> bool:
        """Verify RSA PKCS#1 v1.5 signature"""
        try:
            public_key_pem = self.public_keys[key_id]
            public_key = load_pem_public_key(public_key_pem)
            
            public_key.verify(
                signature,
                data,
                padding.PKCS1v15(),
                hashes.SHA256()
            )
            return True
        except Exception:
            return False
    
    def _sign_hmac(self, data: bytes, key_id: str) -> bytes:
        """Sign data using HMAC-SHA256"""
        hmac_key = self.hmac_keys[key_id]
        signature = hmac.new(hmac_key, data, hashlib.sha256).digest()
        return signature
    
    def _verify_hmac(self, data: bytes, signature: bytes, key_id: str) -> bool:
        """Verify HMAC-SHA256 signature"""
        try:
            hmac_key = self.hmac_keys[key_id]
            expected_signature = hmac.new(hmac_key, data, hashlib.sha256).digest()
            return hmac.compare_digest(signature, expected_signature)
        except Exception:
            return False
    
    def _determine_trust_level(self, environment: SigningEnvironment, 
                             algorithm: SignatureAlgorithm) -> str:
        """Determine trust level based on environment and algorithm"""
        if environment == SigningEnvironment.PRODUCTION and algorithm == SignatureAlgorithm.RSA_PSS_SHA256:
            return "high"
        elif environment in [SigningEnvironment.STAGING, SigningEnvironment.PRODUCTION]:
            return "medium"
        else:
            return "low"

# API Interface
class DigitalSignatureAPI:
    """API interface for digital signature operations"""
    
    def __init__(self, signature_service: Optional[DigitalSignatureService] = None):
        self.signature_service = signature_service or DigitalSignatureService()
    
    def sign_manifest(self, manifest_hash: str, key_id: str, 
                     environment: str, signer_metadata: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """API endpoint to sign a manifest"""
        try:
            env = SigningEnvironment(environment.lower())
            signature = self.signature_service.sign_manifest(
                manifest_hash, key_id, env, signer_metadata
            )
            
            return {
                'success': True,
                'signature_id': signature.signature_id,
                'manifest_hash': signature.manifest_hash,
                'signature_bytes': signature.signature_bytes,
                'algorithm': signature.algorithm.value,
                'key_id': signature.key_id,
                'environment': signature.environment.value,
                'signed_at': signature.signed_at.isoformat()
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': str(e),
                'error_type': type(e).__name__
            }
    
    def verify_signature(self, signature_id: str, manifest_hash: str) -> Dict[str, Any]:
        """API endpoint to verify a signature"""
        result = self.signature_service.verify_signature(signature_id, manifest_hash)
        
        return {
            'success': True,
            'is_valid': result.is_valid,
            'signature_id': result.signature_id,
            'verified_at': result.verified_at.isoformat(),
            'key_id': result.key_id,
            'algorithm': result.algorithm.value,
            'trust_level': result.trust_level,
            'error_message': result.error_message
        }
    
    def validate_policy(self, manifest_hash: str, environment: str) -> Dict[str, Any]:
        """API endpoint to validate signature policy"""
        try:
            env = SigningEnvironment(environment.lower())
            result = self.signature_service.validate_signature_policy(manifest_hash, env)
            return {
                'success': True,
                **result
            }
        except Exception as e:
            return {
                'success': False,
                'error': str(e)
            }

# Test Functions
def run_digital_signature_tests():
    """Run comprehensive digital signature tests"""
    print("=== Digital Signature Service Tests ===")
    
    # Initialize service
    signature_service = DigitalSignatureService()
    signature_api = DigitalSignatureAPI(signature_service)
    
    # Test manifest hash (from previous task)
    test_manifest_hash = "a1b2c3d4e5f6789012345678901234567890abcdef1234567890abcdef123456"
    
    print(f"\nTest manifest hash: {test_manifest_hash[:16]}...")
    
    # Test 1: Generate signing keys
    print("\n1. Testing signing key generation...")
    
    # Generate keys for different environments
    dev_key = signature_service.generate_signing_key(
        "dev_key_001", SignatureAlgorithm.HMAC_SHA256, SigningEnvironment.DEVELOPMENT
    )
    print(f"   Dev key: {dev_key.key_id} ({dev_key.algorithm.value})")
    
    staging_key = signature_service.generate_signing_key(
        "staging_key_001", SignatureAlgorithm.RSA_PKCS1_SHA256, SigningEnvironment.STAGING
    )
    print(f"   Staging key: {staging_key.key_id} ({staging_key.algorithm.value})")
    
    prod_key = signature_service.generate_signing_key(
        "prod_key_001", SignatureAlgorithm.RSA_PSS_SHA256, SigningEnvironment.PRODUCTION
    )
    print(f"   Prod key: {prod_key.key_id} ({prod_key.algorithm.value})")
    
    # Test 2: Sign manifest with different keys
    print("\n2. Testing manifest signing...")
    
    # Sign with dev key
    dev_signature = signature_service.sign_manifest(
        test_manifest_hash, "dev_key_001", SigningEnvironment.DEVELOPMENT,
        {"author": "dev_user", "build_id": "dev_123"}
    )
    print(f"   Dev signature: {dev_signature.signature_id}")
    
    # Sign with staging key
    staging_signature = signature_service.sign_manifest(
        test_manifest_hash, "staging_key_001", SigningEnvironment.STAGING,
        {"author": "staging_user", "build_id": "staging_456"}
    )
    print(f"   Staging signature: {staging_signature.signature_id}")
    
    # Sign with prod key
    prod_signature = signature_service.sign_manifest(
        test_manifest_hash, "prod_key_001", SigningEnvironment.PRODUCTION,
        {"author": "prod_user", "build_id": "prod_789"}
    )
    print(f"   Prod signature: {prod_signature.signature_id}")
    
    # Test 3: Verify signatures
    print("\n3. Testing signature verification...")
    
    # Verify each signature
    signatures_to_test = [
        ("Dev", dev_signature.signature_id),
        ("Staging", staging_signature.signature_id),
        ("Prod", prod_signature.signature_id)
    ]
    
    for name, sig_id in signatures_to_test:
        result = signature_service.verify_signature(sig_id, test_manifest_hash)
        status = "✅ PASS" if result.is_valid else "❌ FAIL"
        print(f"   {name} verification: {status} (trust: {result.trust_level})")
        if not result.is_valid:
            print(f"     Error: {result.error_message}")
    
    # Test 4: Policy validation
    print("\n4. Testing signature policy validation...")
    
    environments = [SigningEnvironment.DEVELOPMENT, SigningEnvironment.STAGING, SigningEnvironment.PRODUCTION]
    for env in environments:
        policy_result = signature_service.validate_signature_policy(test_manifest_hash, env)
        status = "✅ PASS" if policy_result['is_valid'] else "❌ FAIL"
        print(f"   {env.value} policy: {status}")
        if not policy_result['is_valid']:
            print(f"     Reason: {policy_result['reason']}")
    
    # Test 5: API interface
    print("\n5. Testing API interface...")
    
    # Test API signing
    api_sign_result = signature_api.sign_manifest(
        test_manifest_hash, "dev_key_001", "development",
        {"api_test": True}
    )
    print(f"   API signing: {'✅ PASS' if api_sign_result['success'] else '❌ FAIL'}")
    
    if api_sign_result['success']:
        # Test API verification
        api_verify_result = signature_api.verify_signature(
            api_sign_result['signature_id'], test_manifest_hash
        )
        print(f"   API verification: {'✅ PASS' if api_verify_result['is_valid'] else '❌ FAIL'}")
    
    # Test 6: Signature tampering detection
    print("\n6. Testing tampering detection...")
    
    # Try to verify with wrong hash
    tampered_hash = "tampered_hash_123456789012345678901234567890abcdef123456789"
    tamper_result = signature_service.verify_signature(dev_signature.signature_id, tampered_hash)
    tamper_detected = not tamper_result.is_valid
    print(f"   Tampering detection: {'✅ PASS' if tamper_detected else '❌ FAIL'}")
    if tamper_detected:
        print(f"     Error: {tamper_result.error_message}")
    
    # Test 7: Statistics
    print("\n7. Testing statistics...")
    stats = signature_service.get_signature_statistics()
    print(f"   Total signatures created: {stats['total_signatures_created']}")
    print(f"   Total verifications: {stats['total_verifications']}")
    print(f"   Success rate: {stats['success_rate']:.1f}%")
    print(f"   Active keys: {stats['active_keys']}")
    
    # Test 8: Multiple signatures per manifest
    print("\n8. Testing multiple signatures per manifest...")
    
    # Add another signature to the same manifest
    additional_signature = signature_service.sign_manifest(
        test_manifest_hash, "staging_key_001", SigningEnvironment.STAGING,
        {"additional": True}
    )
    
    all_signatures = signature_service.get_manifest_signatures(test_manifest_hash)
    print(f"   Total signatures for manifest: {len(all_signatures)}")
    print(f"   Signature environments: {[sig.environment.value for sig in all_signatures]}")
    
    print(f"\n=== Test Summary ===")
    print(f"Digital signature service tested successfully")
    print(f"Keys generated: {stats['total_signing_keys']}")
    print(f"Signatures created: {stats['total_signatures_created']}")
    print(f"Verifications performed: {stats['total_verifications']}")
    print(f"Success rate: {stats['success_rate']:.1f}%")
    
    return signature_service, signature_api

if __name__ == "__main__":
    run_digital_signature_tests()
