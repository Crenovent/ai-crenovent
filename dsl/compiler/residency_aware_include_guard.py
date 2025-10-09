"""
Residency-Aware Include Guard - Task 6.2.55
===========================================

Import libs only if region-allowed
- Validates imports against region matrix
- Blocks imports from non-allowed regions
- Ensures compliance with data residency requirements
"""

from typing import Dict, List, Any, Optional, Set
import logging
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)

class ResidencyViolationType(Enum):
    BLOCKED_REGION = "blocked_region"
    MISSING_REGION_INFO = "missing_region_info"
    INVALID_REGION = "invalid_region"

@dataclass
class ResidencyViolation:
    violation_type: ResidencyViolationType
    import_path: str
    source_region: Optional[str]
    target_region: str
    message: str

class ResidencyAwareIncludeGuard:
    """Guards imports based on region compatibility matrix"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.region_matrix = self._load_region_matrix()
        
    def _load_region_matrix(self) -> Dict[str, Dict[str, bool]]:
        """Load region compatibility matrix"""
        return {
            'us-east-1': {
                'us-east-1': True, 'us-west-2': True, 'eu-west-1': False, 'ap-southeast-1': False
            },
            'eu-west-1': {
                'eu-west-1': True, 'eu-central-1': True, 'us-east-1': False, 'us-west-2': False
            },
            'ap-southeast-1': {
                'ap-southeast-1': True, 'ap-northeast-1': True, 'us-east-1': False, 'eu-west-1': False
            }
        }
    
    def validate_import(self, import_path: str, source_region: str, 
                       target_region: Optional[str] = None) -> List[ResidencyViolation]:
        """Validate import against region rules"""
        violations = []
        
        if not target_region:
            violations.append(ResidencyViolation(
                ResidencyViolationType.MISSING_REGION_INFO,
                import_path, source_region, "unknown",
                f"Import {import_path} missing region information"
            ))
            return violations
        
        if source_region not in self.region_matrix:
            violations.append(ResidencyViolation(
                ResidencyViolationType.INVALID_REGION,
                import_path, source_region, target_region,
                f"Source region {source_region} not in compatibility matrix"
            ))
            return violations
        
        allowed = self.region_matrix[source_region].get(target_region, False)
        if not allowed:
            violations.append(ResidencyViolation(
                ResidencyViolationType.BLOCKED_REGION,
                import_path, source_region, target_region,
                f"Import from {target_region} blocked in {source_region}"
            ))
        
        return violations
    
    def filter_allowed_imports(self, imports: List[Dict[str, Any]], 
                              source_region: str) -> List[Dict[str, Any]]:
        """Filter imports to only allowed ones"""
        allowed_imports = []
        
        for import_item in imports:
            import_path = import_item.get('path', '')
            target_region = import_item.get('region')
            
            violations = self.validate_import(import_path, source_region, target_region)
            if not violations:
                allowed_imports.append(import_item)
            else:
                self.logger.warning(f"Blocked import: {import_path} - {violations[0].message}")
        
        return allowed_imports
