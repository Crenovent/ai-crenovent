"""
RBIA Plan Hash Service - Task 6.1.20
=====================================
Generate immutable hashes for RBIA execution plans

Hashes include:
- Input parameters
- Model version
- Confidence thresholds
- Policy pack ID
- Workflow structure
"""

import hashlib
import json
from typing import Dict, List, Any, Optional
from datetime import datetime
from dataclasses import dataclass, field
import logging

logger = logging.getLogger(__name__)


@dataclass
class PlanHashSpec:
    """Specification for RBIA plan hashing"""
    plan_id: str
    workflow_id: str
    tenant_id: str
    
    # Workflow components
    workflow_structure: Dict[str, Any]
    input_parameters: Dict[str, Any]
    
    # ML components
    model_versions: Dict[str, str]  # model_id -> version
    confidence_thresholds: Dict[str, float]  # node_id -> threshold
    
    # Governance
    policy_pack_id: Optional[str] = None
    compliance_requirements: List[str] = field(default_factory=list)
    
    # Metadata
    created_at: datetime = field(default_factory=datetime.utcnow)
    created_by: str = "system"
    
    # Hash result
    plan_hash: Optional[str] = None
    hash_algorithm: str = "SHA-256"


class PlanHashService:
    """Service for generating and validating RBIA plan hashes"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.hash_registry: Dict[str, PlanHashSpec] = {}
    
    def generate_plan_hash(self, plan_spec: PlanHashSpec) -> str:
        """
        Generate deterministic hash for an RBIA execution plan
        
        Args:
            plan_spec: Plan hash specification
            
        Returns:
            SHA-256 hash of the plan
        """
        # Create canonical representation
        hash_input = {
            'workflow_id': plan_spec.workflow_id,
            'workflow_structure': self._canonicalize(plan_spec.workflow_structure),
            'input_parameters': self._canonicalize(plan_spec.input_parameters),
            'model_versions': self._canonicalize(plan_spec.model_versions),
            'confidence_thresholds': self._canonicalize(plan_spec.confidence_thresholds),
            'policy_pack_id': plan_spec.policy_pack_id,
            'compliance_requirements': sorted(plan_spec.compliance_requirements)
        }
        
        # Convert to JSON with sorted keys
        canonical_json = json.dumps(hash_input, sort_keys=True, separators=(',', ':'))
        
        # Generate hash
        hash_object = hashlib.sha256(canonical_json.encode('utf-8'))
        plan_hash = hash_object.hexdigest()
        
        # Store hash
        plan_spec.plan_hash = plan_hash
        self.hash_registry[plan_hash] = plan_spec
        
        self.logger.info(f"✅ Generated plan hash: {plan_hash[:16]}...")
        
        return plan_hash
    
    def verify_plan_hash(self, plan_spec: PlanHashSpec, expected_hash: str) -> bool:
        """Verify plan hash matches expected value"""
        actual_hash = self.generate_plan_hash(plan_spec)
        return actual_hash == expected_hash
    
    def get_plan_by_hash(self, plan_hash: str) -> Optional[PlanHashSpec]:
        """Retrieve plan specification by hash"""
        return self.hash_registry.get(plan_hash)
    
    def _canonicalize(self, obj: Any) -> Any:
        """Convert object to canonical form for hashing"""
        if isinstance(obj, dict):
            return {k: self._canonicalize(v) for k, v in sorted(obj.items())}
        elif isinstance(obj, list):
            return [self._canonicalize(item) for item in obj]
        elif isinstance(obj, (int, float, str, bool, type(None))):
            return obj
        else:
            return str(obj)


# Singleton
_plan_hash_service: Optional[PlanHashService] = None

def get_plan_hash_service() -> PlanHashService:
    global _plan_hash_service
    if _plan_hash_service is None:
        _plan_hash_service = PlanHashService()
    return _plan_hash_service

