"""
Plan Hash Service - Task 6.2.27
================================

Plan hash (SHA-256 over canonical manifest)
- Creates immutable, content-addressed identifier for each manifest
- Computes SHA-256 over canonical compact JSON from 6.2.26
- Provides API to output manifest hash and verify manifests
- Backend storage mapping for manifest → hash
- Determinism verification tests

Dependencies: Task 6.2.26 (Plan Manifest Generator)
Outputs: Immutable plan hash → enables content-addressed storage and verification
"""

from typing import Dict, List, Any, Optional, Tuple
import hashlib
import json
import logging
from datetime import datetime
from dataclasses import dataclass, field
from enum import Enum
import time

logger = logging.getLogger(__name__)

class HashAlgorithm(Enum):
    """Supported hash algorithms"""
    SHA256 = "sha256"
    SHA512 = "sha512"
    BLAKE2B = "blake2b"

@dataclass
class PlanHashResult:
    """Result of plan hash computation"""
    plan_id: str
    manifest_hash: str
    algorithm: HashAlgorithm
    computed_at: datetime
    manifest_size_bytes: int
    computation_time_ms: float
    verification_status: str = "valid"
    
@dataclass
class HashVerificationResult:
    """Result of hash verification"""
    is_valid: bool
    expected_hash: str
    actual_hash: str
    algorithm: HashAlgorithm
    verified_at: datetime
    error_message: Optional[str] = None

@dataclass
class ManifestHashMapping:
    """Mapping between manifest content and hash"""
    manifest_hash: str
    plan_id: str
    workflow_id: str
    manifest_size_bytes: int
    algorithm: HashAlgorithm
    created_at: datetime
    last_verified_at: Optional[datetime] = None
    verification_count: int = 0
    content_preview: str = ""  # First 200 chars of canonical manifest

# Task 6.2.27: Plan Hash Service
class PlanHashService:
    """Service for computing and verifying plan manifest hashes"""
    
    def __init__(self, default_algorithm: HashAlgorithm = HashAlgorithm.SHA256):
        self.logger = logging.getLogger(__name__)
        self.default_algorithm = default_algorithm
        
        # In-memory storage for hash mappings (backend implementation)
        self.hash_mappings: Dict[str, ManifestHashMapping] = {}
        self.plan_to_hash: Dict[str, str] = {}  # plan_id -> hash
        
        # Hash computation statistics
        self.computation_stats = {
            'total_hashes_computed': 0,
            'total_verifications': 0,
            'verification_failures': 0,
            'average_computation_time_ms': 0.0
        }
    
    def compute_manifest_hash(self, canonical_manifest: str, 
                            plan_id: str = "",
                            algorithm: Optional[HashAlgorithm] = None) -> PlanHashResult:
        """
        Compute SHA-256 (or specified) hash over canonical manifest
        
        Args:
            canonical_manifest: Canonical compact JSON from plan manifest generator
            plan_id: Plan identifier for tracking
            algorithm: Hash algorithm to use (defaults to SHA256)
            
        Returns:
            PlanHashResult with hash and metadata
        """
        start_time = time.time()
        algorithm = algorithm or self.default_algorithm
        
        try:
            # Validate manifest is valid JSON
            json.loads(canonical_manifest)
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON manifest: {e}")
        
        # Compute hash based on algorithm
        manifest_bytes = canonical_manifest.encode('utf-8')
        
        if algorithm == HashAlgorithm.SHA256:
            hash_obj = hashlib.sha256(manifest_bytes)
        elif algorithm == HashAlgorithm.SHA512:
            hash_obj = hashlib.sha512(manifest_bytes)
        elif algorithm == HashAlgorithm.BLAKE2B:
            hash_obj = hashlib.blake2b(manifest_bytes)
        else:
            raise ValueError(f"Unsupported hash algorithm: {algorithm}")
        
        manifest_hash = hash_obj.hexdigest()
        
        # Calculate computation time
        computation_time_ms = (time.time() - start_time) * 1000
        
        # Create result
        result = PlanHashResult(
            plan_id=plan_id,
            manifest_hash=manifest_hash,
            algorithm=algorithm,
            computed_at=datetime.utcnow(),
            manifest_size_bytes=len(manifest_bytes),
            computation_time_ms=computation_time_ms
        )
        
        # Store mapping in backend storage
        self._store_hash_mapping(canonical_manifest, result)
        
        # Update statistics
        self._update_computation_stats(computation_time_ms)
        
        self.logger.info(f"✅ Computed {algorithm.value} hash for plan {plan_id}: {manifest_hash[:16]}...")
        
        return result
    
    def verify_manifest_hash(self, canonical_manifest: str, 
                           expected_hash: str,
                           algorithm: Optional[HashAlgorithm] = None) -> HashVerificationResult:
        """
        Verify manifest by recomputing hash and comparing
        
        Args:
            canonical_manifest: Canonical manifest to verify
            expected_hash: Expected hash value
            algorithm: Hash algorithm used
            
        Returns:
            HashVerificationResult with verification status
        """
        algorithm = algorithm or self.default_algorithm
        
        try:
            # Recompute hash
            result = self.compute_manifest_hash(canonical_manifest, algorithm=algorithm)
            actual_hash = result.manifest_hash
            
            # Compare hashes
            is_valid = actual_hash == expected_hash
            
            verification_result = HashVerificationResult(
                is_valid=is_valid,
                expected_hash=expected_hash,
                actual_hash=actual_hash,
                algorithm=algorithm,
                verified_at=datetime.utcnow()
            )
            
            if not is_valid:
                verification_result.error_message = f"Hash mismatch: expected {expected_hash[:16]}..., got {actual_hash[:16]}..."
                self.computation_stats['verification_failures'] += 1
                self.logger.warning(f"❌ Hash verification failed: {verification_result.error_message}")
            else:
                self.logger.info(f"✅ Hash verification passed: {expected_hash[:16]}...")
            
            # Update verification statistics
            self.computation_stats['total_verifications'] += 1
            
            # Update mapping verification info
            if expected_hash in self.hash_mappings:
                mapping = self.hash_mappings[expected_hash]
                mapping.last_verified_at = datetime.utcnow()
                mapping.verification_count += 1
            
            return verification_result
            
        except Exception as e:
            return HashVerificationResult(
                is_valid=False,
                expected_hash=expected_hash,
                actual_hash="",
                algorithm=algorithm,
                verified_at=datetime.utcnow(),
                error_message=f"Verification error: {str(e)}"
            )
    
    def get_manifest_by_hash(self, manifest_hash: str) -> Optional[ManifestHashMapping]:
        """
        Retrieve manifest mapping by hash (content-addressed lookup)
        
        Args:
            manifest_hash: Hash to look up
            
        Returns:
            ManifestHashMapping if found, None otherwise
        """
        return self.hash_mappings.get(manifest_hash)
    
    def get_hash_by_plan_id(self, plan_id: str) -> Optional[str]:
        """
        Get hash for a specific plan ID
        
        Args:
            plan_id: Plan identifier
            
        Returns:
            Hash string if found, None otherwise
        """
        return self.plan_to_hash.get(plan_id)
    
    def list_all_hashes(self, limit: Optional[int] = None) -> List[ManifestHashMapping]:
        """
        List all stored hash mappings
        
        Args:
            limit: Maximum number of results to return
            
        Returns:
            List of ManifestHashMapping objects
        """
        mappings = list(self.hash_mappings.values())
        mappings.sort(key=lambda x: x.created_at, reverse=True)
        
        if limit:
            mappings = mappings[:limit]
        
        return mappings
    
    def verify_hash_determinism(self, canonical_manifest: str, 
                              iterations: int = 5) -> Dict[str, Any]:
        """
        Test hash determinism by computing hash multiple times
        
        Args:
            canonical_manifest: Manifest to test
            iterations: Number of iterations to run
            
        Returns:
            Dictionary with determinism test results
        """
        hashes = []
        computation_times = []
        
        for i in range(iterations):
            result = self.compute_manifest_hash(canonical_manifest, plan_id=f"determinism_test_{i}")
            hashes.append(result.manifest_hash)
            computation_times.append(result.computation_time_ms)
        
        # Check if all hashes are identical
        is_deterministic = all(h == hashes[0] for h in hashes)
        
        return {
            'is_deterministic': is_deterministic,
            'unique_hashes': len(set(hashes)),
            'expected_unique_hashes': 1,
            'sample_hash': hashes[0] if hashes else None,
            'iterations': iterations,
            'average_computation_time_ms': sum(computation_times) / len(computation_times),
            'min_computation_time_ms': min(computation_times),
            'max_computation_time_ms': max(computation_times),
            'all_hashes_identical': is_deterministic
        }
    
    def get_computation_statistics(self) -> Dict[str, Any]:
        """Get hash computation and verification statistics"""
        return {
            **self.computation_stats,
            'total_stored_mappings': len(self.hash_mappings),
            'success_rate': (
                (self.computation_stats['total_verifications'] - self.computation_stats['verification_failures']) 
                / max(1, self.computation_stats['total_verifications'])
            ) * 100
        }
    
    def _store_hash_mapping(self, canonical_manifest: str, result: PlanHashResult):
        """Store hash mapping in backend storage"""
        try:
            # Extract workflow_id from manifest
            manifest_data = json.loads(canonical_manifest)
            workflow_id = manifest_data.get('workflow_id', 'unknown')
            
            # Create mapping
            mapping = ManifestHashMapping(
                manifest_hash=result.manifest_hash,
                plan_id=result.plan_id,
                workflow_id=workflow_id,
                manifest_size_bytes=result.manifest_size_bytes,
                algorithm=result.algorithm,
                created_at=result.computed_at,
                content_preview=canonical_manifest[:200] + ("..." if len(canonical_manifest) > 200 else "")
            )
            
            # Store mappings
            self.hash_mappings[result.manifest_hash] = mapping
            if result.plan_id:
                self.plan_to_hash[result.plan_id] = result.manifest_hash
            
        except Exception as e:
            self.logger.warning(f"Failed to store hash mapping: {e}")
    
    def _update_computation_stats(self, computation_time_ms: float):
        """Update computation statistics"""
        self.computation_stats['total_hashes_computed'] += 1
        
        # Update rolling average
        total = self.computation_stats['total_hashes_computed']
        current_avg = self.computation_stats['average_computation_time_ms']
        new_avg = ((current_avg * (total - 1)) + computation_time_ms) / total
        self.computation_stats['average_computation_time_ms'] = new_avg

# API/CLI Interface Functions
class PlanHashAPI:
    """API interface for plan hash operations"""
    
    def __init__(self, hash_service: Optional[PlanHashService] = None):
        self.hash_service = hash_service or PlanHashService()
    
    def hash_manifest(self, canonical_manifest: str, 
                     plan_id: str = "",
                     algorithm: str = "sha256") -> Dict[str, Any]:
        """
        API endpoint to compute manifest hash
        
        Args:
            canonical_manifest: Canonical JSON manifest
            plan_id: Plan identifier
            algorithm: Hash algorithm (sha256, sha512, blake2b)
            
        Returns:
            Dictionary with hash result
        """
        try:
            hash_algorithm = HashAlgorithm(algorithm.lower())
            result = self.hash_service.compute_manifest_hash(
                canonical_manifest, plan_id, hash_algorithm
            )
            
            return {
                'success': True,
                'plan_id': result.plan_id,
                'manifest_hash': result.manifest_hash,
                'algorithm': result.algorithm.value,
                'computed_at': result.computed_at.isoformat(),
                'manifest_size_bytes': result.manifest_size_bytes,
                'computation_time_ms': result.computation_time_ms
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': str(e),
                'error_type': type(e).__name__
            }
    
    def verify_manifest(self, canonical_manifest: str, 
                       expected_hash: str,
                       algorithm: str = "sha256") -> Dict[str, Any]:
        """
        API endpoint to verify manifest hash
        
        Args:
            canonical_manifest: Canonical JSON manifest
            expected_hash: Expected hash value
            algorithm: Hash algorithm used
            
        Returns:
            Dictionary with verification result
        """
        try:
            hash_algorithm = HashAlgorithm(algorithm.lower())
            result = self.hash_service.verify_manifest_hash(
                canonical_manifest, expected_hash, hash_algorithm
            )
            
            return {
                'success': True,
                'is_valid': result.is_valid,
                'expected_hash': result.expected_hash,
                'actual_hash': result.actual_hash,
                'algorithm': result.algorithm.value,
                'verified_at': result.verified_at.isoformat(),
                'error_message': result.error_message
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': str(e),
                'error_type': type(e).__name__
            }
    
    def lookup_manifest(self, manifest_hash: str) -> Dict[str, Any]:
        """
        API endpoint to lookup manifest by hash
        
        Args:
            manifest_hash: Hash to look up
            
        Returns:
            Dictionary with manifest mapping info
        """
        mapping = self.hash_service.get_manifest_by_hash(manifest_hash)
        
        if mapping:
            return {
                'success': True,
                'found': True,
                'manifest_hash': mapping.manifest_hash,
                'plan_id': mapping.plan_id,
                'workflow_id': mapping.workflow_id,
                'manifest_size_bytes': mapping.manifest_size_bytes,
                'algorithm': mapping.algorithm.value,
                'created_at': mapping.created_at.isoformat(),
                'last_verified_at': mapping.last_verified_at.isoformat() if mapping.last_verified_at else None,
                'verification_count': mapping.verification_count,
                'content_preview': mapping.content_preview
            }
        else:
            return {
                'success': True,
                'found': False,
                'manifest_hash': manifest_hash
            }
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get hash service statistics"""
        return {
            'success': True,
            'statistics': self.hash_service.get_computation_statistics()
        }

# Test Functions
def create_test_canonical_manifest() -> str:
    """Create a test canonical manifest for testing"""
    test_manifest = {
        "manifest_version": "1.0.0",
        "plan_id": "test_plan_001",
        "workflow_id": "test_workflow",
        "name": "Test Hash Workflow",
        "version": "1.0.0",
        "automation_type": "rbia",
        "plan_graph": {
            "nodes": [
                {
                    "id": "node_001",
                    "type": "ml_predict",
                    "inputs": [{"name": "customer_data", "type": "object", "required": True}],
                    "outputs": [{"name": "prediction", "type": "float"}],
                    "parameters": {"model_id": "test_model_v1"}
                }
            ],
            "edges": []
        },
        "trust_budgets": {
            "node_001": {
                "min_trust": 0.7,
                "auto_execute_above": 0.85,
                "trust_level": "medium"
            }
        },
        "fallback_dag": {"nodes": {}, "edges": []},
        "explainability_hooks": {},
        "generated_at": "2024-01-01T00:00:00.000000Z"
    }
    
    # Return canonical JSON (sorted keys, compact)
    return json.dumps(test_manifest, sort_keys=True, separators=(',', ':'))

def run_plan_hash_tests():
    """Run comprehensive plan hash tests"""
    print("=== Plan Hash Service Tests ===")
    
    # Initialize service and API
    hash_service = PlanHashService()
    hash_api = PlanHashAPI(hash_service)
    
    # Create test manifest
    canonical_manifest = create_test_canonical_manifest()
    plan_id = "test_plan_001"
    
    print(f"\nTest manifest size: {len(canonical_manifest)} bytes")
    
    # Test 1: Basic hash computation
    print("\n1. Testing basic hash computation...")
    result = hash_service.compute_manifest_hash(canonical_manifest, plan_id)
    print(f"   Plan ID: {result.plan_id}")
    print(f"   Hash: {result.manifest_hash}")
    print(f"   Algorithm: {result.algorithm.value}")
    print(f"   Computation time: {result.computation_time_ms:.2f}ms")
    
    # Test 2: Hash verification
    print("\n2. Testing hash verification...")
    verification = hash_service.verify_manifest_hash(canonical_manifest, result.manifest_hash)
    print(f"   Verification result: {'✅ PASS' if verification.is_valid else '❌ FAIL'}")
    if not verification.is_valid:
        print(f"   Error: {verification.error_message}")
    
    # Test 3: Determinism verification
    print("\n3. Testing hash determinism...")
    determinism_result = hash_service.verify_hash_determinism(canonical_manifest, iterations=5)
    print(f"   Deterministic: {'✅ PASS' if determinism_result['is_deterministic'] else '❌ FAIL'}")
    print(f"   Unique hashes: {determinism_result['unique_hashes']} (expected: 1)")
    print(f"   Average computation time: {determinism_result['average_computation_time_ms']:.2f}ms")
    
    # Test 4: Different algorithms
    print("\n4. Testing different hash algorithms...")
    algorithms = [HashAlgorithm.SHA256, HashAlgorithm.SHA512, HashAlgorithm.BLAKE2B]
    for algo in algorithms:
        result = hash_service.compute_manifest_hash(canonical_manifest, f"{plan_id}_{algo.value}", algo)
        print(f"   {algo.value}: {result.manifest_hash[:16]}... ({result.computation_time_ms:.2f}ms)")
    
    # Test 5: Content-addressed lookup
    print("\n5. Testing content-addressed lookup...")
    original_hash = hash_service.compute_manifest_hash(canonical_manifest, plan_id).manifest_hash
    mapping = hash_service.get_manifest_by_hash(original_hash)
    if mapping:
        print(f"   Found mapping: plan_id={mapping.plan_id}, size={mapping.manifest_size_bytes} bytes")
        print(f"   Preview: {mapping.content_preview[:50]}...")
    else:
        print("   ❌ Mapping not found")
    
    # Test 6: API interface
    print("\n6. Testing API interface...")
    api_result = hash_api.hash_manifest(canonical_manifest, plan_id)
    print(f"   API hash result: {'✅ PASS' if api_result['success'] else '❌ FAIL'}")
    if api_result['success']:
        print(f"   API hash: {api_result['manifest_hash'][:16]}...")
    
    api_verification = hash_api.verify_manifest(canonical_manifest, original_hash)
    print(f"   API verification: {'✅ PASS' if api_verification['is_valid'] else '❌ FAIL'}")
    
    # Test 7: Statistics
    print("\n7. Testing statistics...")
    stats = hash_service.get_computation_statistics()
    print(f"   Total hashes computed: {stats['total_hashes_computed']}")
    print(f"   Total verifications: {stats['total_verifications']}")
    print(f"   Success rate: {stats['success_rate']:.1f}%")
    print(f"   Average computation time: {stats['average_computation_time_ms']:.2f}ms")
    
    # Test 8: Hash changes with content changes
    print("\n8. Testing hash sensitivity to content changes...")
    # Modify manifest slightly
    modified_manifest_data = json.loads(canonical_manifest)
    modified_manifest_data['version'] = '1.0.1'  # Small change
    modified_manifest = json.dumps(modified_manifest_data, sort_keys=True, separators=(',', ':'))
    
    original_hash = hash_service.compute_manifest_hash(canonical_manifest, "original").manifest_hash
    modified_hash = hash_service.compute_manifest_hash(modified_manifest, "modified").manifest_hash
    
    hashes_different = original_hash != modified_hash
    print(f"   Hash changes with content: {'✅ PASS' if hashes_different else '❌ FAIL'}")
    print(f"   Original: {original_hash[:16]}...")
    print(f"   Modified: {modified_hash[:16]}...")
    
    print(f"\n=== Test Summary ===")
    print(f"Hash service initialized and tested successfully")
    print(f"Sample hash: {original_hash}")
    print(f"Total operations: {stats['total_hashes_computed']} computations, {stats['total_verifications']} verifications")
    
    return hash_service, hash_api

if __name__ == "__main__":
    run_plan_hash_tests()
