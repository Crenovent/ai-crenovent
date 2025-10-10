# Multi-Industry Adapter SDK - Chapter 9.1
# Tasks 9.1-T01 to T60: Core reusable backbone with industry overlays

import json
import uuid
import hashlib
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Any, Optional, Union, Protocol, runtime_checkable
from dataclasses import dataclass, asdict, field
from enum import Enum
import logging
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)

class IndustryType(Enum):
    """Supported industry types"""
    SAAS = "saas"
    BANKING = "banking"
    INSURANCE = "insurance"
    ECOMMERCE = "ecommerce"
    FINANCIAL_SERVICES = "financial_services"
    IT_SERVICES = "it_services"
    HEALTHCARE = "healthcare"
    MANUFACTURING = "manufacturing"

class OverlayType(Enum):
    """Types of overlays"""
    POLICY = "policy"
    SCHEMA = "schema"
    DSL_MACRO = "dsl_macro"
    CONNECTOR = "connector"
    DASHBOARD = "dashboard"
    COMPLIANCE = "compliance"
    PARAMETER = "parameter"

class OverlayStatus(Enum):
    """Overlay lifecycle status"""
    DRAFT = "draft"
    REVIEW = "review"
    APPROVED = "approved"
    PUBLISHED = "published"
    DEPRECATED = "deprecated"
    ARCHIVED = "archived"

@dataclass
class OverlayManifest:
    """
    Overlay Manifest Specification - Tasks 9.1-T04, T05
    Deterministic packaging with versioning, dependencies, and governance
    """
    # Core identification
    overlay_id: str
    name: str
    version: str
    industry: IndustryType
    overlay_type: OverlayType
    
    # Metadata
    description: str
    created_by: str
    created_at: str
    
    # Dependencies and compatibility
    dependencies: List[str] = field(default_factory=list)
    backbone_compatibility: str = ">=1.0.0"
    
    # Governance and compliance
    residency_requirements: List[str] = field(default_factory=list)  # ["EU", "US", "IN"]
    sod_requirements: List[str] = field(default_factory=list)  # ["maker_checker", "dual_approval"]
    compliance_frameworks: List[str] = field(default_factory=list)  # ["SOX", "GDPR", "RBI"]
    
    # Technical specifications
    schema_extensions: Dict[str, Any] = field(default_factory=dict)
    policy_rules: Dict[str, Any] = field(default_factory=dict)
    dsl_macros: Dict[str, Any] = field(default_factory=dict)
    connector_configs: Dict[str, Any] = field(default_factory=dict)
    
    # Lifecycle and status
    status: OverlayStatus = OverlayStatus.DRAFT
    
    # Provenance and attestation
    content_hash: Optional[str] = None
    signature: Optional[str] = None
    sbom: Optional[Dict[str, Any]] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        data = asdict(self)
        data['industry'] = self.industry.value
        data['overlay_type'] = self.overlay_type.value
        data['status'] = self.status.value
        return data
    
    def calculate_content_hash(self) -> str:
        """Calculate SHA256 hash of overlay content"""
        hash_content = {
            'overlay_id': self.overlay_id,
            'name': self.name,
            'version': self.version,
            'industry': self.industry.value,
            'overlay_type': self.overlay_type.value,
            'schema_extensions': self.schema_extensions,
            'policy_rules': self.policy_rules,
            'dsl_macros': self.dsl_macros,
            'connector_configs': self.connector_configs
        }
        
        content_json = json.dumps(hash_content, sort_keys=True, separators=(',', ':'))
        return hashlib.sha256(content_json.encode()).hexdigest()

@runtime_checkable
class PolicyAdapter(Protocol):
    """Policy adapter interface for industry overlays"""
    
    def validate_policy(self, policy_data: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
        """Validate policy against industry requirements"""
        ...
    
    def apply_policy_overlay(self, base_policy: Dict[str, Any], overlay: Dict[str, Any]) -> Dict[str, Any]:
        """Apply industry overlay to base policy"""
        ...

@runtime_checkable
class SchemaAdapter(Protocol):
    """Schema adapter interface for industry overlays"""
    
    def extend_schema(self, base_schema: Dict[str, Any], extensions: Dict[str, Any]) -> Dict[str, Any]:
        """Extend base schema with industry-specific fields"""
        ...
    
    def validate_schema_compatibility(self, base_schema: Dict[str, Any], overlay_schema: Dict[str, Any]) -> bool:
        """Validate schema compatibility"""
        ...

@runtime_checkable
class DSLMacroAdapter(Protocol):
    """DSL macro adapter interface for industry overlays"""
    
    def register_macros(self, macros: Dict[str, Any]) -> bool:
        """Register industry-specific DSL macros"""
        ...
    
    def expand_macro(self, macro_name: str, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Expand DSL macro with parameters"""
        ...

@runtime_checkable
class ConnectorAdapter(Protocol):
    """Connector adapter interface for industry overlays"""
    
    def create_connector_shim(self, connector_type: str, config: Dict[str, Any]) -> Any:
        """Create industry-normalized connector shim"""
        ...
    
    def validate_connector_config(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Validate connector configuration"""
        ...

@runtime_checkable
class DashboardAdapter(Protocol):
    """Dashboard adapter interface for industry overlays"""
    
    def create_dashboard_config(self, dashboard_type: str, kpis: List[str]) -> Dict[str, Any]:
        """Create industry-specific dashboard configuration"""
        ...
    
    def get_industry_kpis(self) -> List[str]:
        """Get industry-specific KPIs"""
        ...

class AdapterSDK:
    """
    Multi-Industry Adapter SDK - Tasks 9.1-T03, T06, T07
    Provides stable extension points for industry overlays
    """
    
    def __init__(self):
        self.policy_adapters: Dict[IndustryType, PolicyAdapter] = {}
        self.schema_adapters: Dict[IndustryType, SchemaAdapter] = {}
        self.dsl_macro_adapters: Dict[IndustryType, DSLMacroAdapter] = {}
        self.connector_adapters: Dict[IndustryType, ConnectorAdapter] = {}
        self.dashboard_adapters: Dict[IndustryType, DashboardAdapter] = {}
        self.logger = logging.getLogger(__name__)
    
    def register_policy_adapter(self, industry: IndustryType, adapter: PolicyAdapter) -> None:
        """Register policy adapter for industry"""
        self.policy_adapters[industry] = adapter
        self.logger.info(f"✅ Registered policy adapter for {industry.value}")
    
    def register_schema_adapter(self, industry: IndustryType, adapter: SchemaAdapter) -> None:
        """Register schema adapter for industry"""
        self.schema_adapters[industry] = adapter
        self.logger.info(f"✅ Registered schema adapter for {industry.value}")
    
    def register_dsl_macro_adapter(self, industry: IndustryType, adapter: DSLMacroAdapter) -> None:
        """Register DSL macro adapter for industry"""
        self.dsl_macro_adapters[industry] = adapter
        self.logger.info(f"✅ Registered DSL macro adapter for {industry.value}")
    
    def register_connector_adapter(self, industry: IndustryType, adapter: ConnectorAdapter) -> None:
        """Register connector adapter for industry"""
        self.connector_adapters[industry] = adapter
        self.logger.info(f"✅ Registered connector adapter for {industry.value}")
    
    def register_dashboard_adapter(self, industry: IndustryType, adapter: DashboardAdapter) -> None:
        """Register dashboard adapter for industry"""
        self.dashboard_adapters[industry] = adapter
        self.logger.info(f"✅ Registered dashboard adapter for {industry.value}")
    
    def get_policy_adapter(self, industry: IndustryType) -> Optional[PolicyAdapter]:
        """Get policy adapter for industry"""
        return self.policy_adapters.get(industry)
    
    def get_schema_adapter(self, industry: IndustryType) -> Optional[SchemaAdapter]:
        """Get schema adapter for industry"""
        return self.schema_adapters.get(industry)
    
    def get_dsl_macro_adapter(self, industry: IndustryType) -> Optional[DSLMacroAdapter]:
        """Get DSL macro adapter for industry"""
        return self.dsl_macro_adapters.get(industry)
    
    def get_connector_adapter(self, industry: IndustryType) -> Optional[ConnectorAdapter]:
        """Get connector adapter for industry"""
        return self.connector_adapters.get(industry)
    
    def get_dashboard_adapter(self, industry: IndustryType) -> Optional[DashboardAdapter]:
        """Get dashboard adapter for industry"""
        return self.dashboard_adapters.get(industry)
    
    def validate_overlay_compatibility(self, overlay: OverlayManifest) -> Dict[str, Any]:
        """Validate overlay compatibility with backbone"""
        validation_result = {
            'compatible': True,
            'issues': [],
            'warnings': []
        }
        
        # Check if adapters are registered for the industry
        industry = overlay.industry
        
        if overlay.overlay_type == OverlayType.POLICY and industry not in self.policy_adapters:
            validation_result['issues'].append(f"No policy adapter registered for {industry.value}")
            validation_result['compatible'] = False
        
        if overlay.overlay_type == OverlayType.SCHEMA and industry not in self.schema_adapters:
            validation_result['issues'].append(f"No schema adapter registered for {industry.value}")
            validation_result['compatible'] = False
        
        if overlay.overlay_type == OverlayType.DSL_MACRO and industry not in self.dsl_macro_adapters:
            validation_result['issues'].append(f"No DSL macro adapter registered for {industry.value}")
            validation_result['compatible'] = False
        
        if overlay.overlay_type == OverlayType.CONNECTOR and industry not in self.connector_adapters:
            validation_result['issues'].append(f"No connector adapter registered for {industry.value}")
            validation_result['compatible'] = False
        
        if overlay.overlay_type == OverlayType.DASHBOARD and industry not in self.dashboard_adapters:
            validation_result['issues'].append(f"No dashboard adapter registered for {industry.value}")
            validation_result['compatible'] = False
        
        # Validate dependencies
        for dep in overlay.dependencies:
            validation_result['warnings'].append(f"Dependency validation not implemented: {dep}")
        
        return validation_result

class OverlayRegistry:
    """
    Overlay Registry - Tasks 9.1-T11, T12, T13
    Manages versioned overlay packs with provenance and attestation
    """
    
    def __init__(self):
        self.overlays: Dict[str, OverlayManifest] = {}
        self.overlay_scores: Dict[str, Dict[str, Any]] = {}
        self.logger = logging.getLogger(__name__)
    
    def register_overlay(self, overlay: OverlayManifest) -> bool:
        """Register overlay in registry"""
        try:
            # Calculate content hash
            overlay.content_hash = overlay.calculate_content_hash()
            
            # Store overlay
            self.overlays[overlay.overlay_id] = overlay
            
            # Initialize scoring
            self.overlay_scores[overlay.overlay_id] = {
                'adoption_score': 0.0,
                'slo_pass_rate': 1.0,
                'incident_count': 0,
                'trust_score': 1.0,
                'last_updated': datetime.now(timezone.utc).isoformat()
            }
            
            self.logger.info(f"✅ Registered overlay: {overlay.overlay_id} ({overlay.industry.value})")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Failed to register overlay {overlay.overlay_id}: {e}")
            return False
    
    def get_overlay(self, overlay_id: str) -> Optional[OverlayManifest]:
        """Get overlay by ID"""
        return self.overlays.get(overlay_id)
    
    def list_overlays(
        self,
        industry: Optional[IndustryType] = None,
        overlay_type: Optional[OverlayType] = None,
        status: Optional[OverlayStatus] = None
    ) -> List[OverlayManifest]:
        """List overlays with filtering"""
        overlays = list(self.overlays.values())
        
        if industry:
            overlays = [o for o in overlays if o.industry == industry]
        
        if overlay_type:
            overlays = [o for o in overlays if o.overlay_type == overlay_type]
        
        if status:
            overlays = [o for o in overlays if o.status == status]
        
        return overlays
    
    def update_overlay_score(
        self,
        overlay_id: str,
        adoption_score: Optional[float] = None,
        slo_pass_rate: Optional[float] = None,
        incident_count: Optional[int] = None
    ) -> None:
        """Update overlay scoring metrics"""
        if overlay_id not in self.overlay_scores:
            return
        
        scores = self.overlay_scores[overlay_id]
        
        if adoption_score is not None:
            scores['adoption_score'] = adoption_score
        
        if slo_pass_rate is not None:
            scores['slo_pass_rate'] = slo_pass_rate
        
        if incident_count is not None:
            scores['incident_count'] = incident_count
        
        # Calculate trust score
        trust_score = (scores['adoption_score'] * 0.3 + 
                      scores['slo_pass_rate'] * 0.5 + 
                      max(0, 1.0 - scores['incident_count'] * 0.1) * 0.2)
        scores['trust_score'] = min(1.0, max(0.0, trust_score))
        scores['last_updated'] = datetime.now(timezone.utc).isoformat()
        
        self.logger.info(f"✅ Updated overlay score for {overlay_id}: trust={trust_score:.3f}")
    
    def get_overlay_score(self, overlay_id: str) -> Optional[Dict[str, Any]]:
        """Get overlay scoring metrics"""
        return self.overlay_scores.get(overlay_id)
    
    def search_overlays(
        self,
        search_term: str,
        industry: Optional[IndustryType] = None
    ) -> List[OverlayManifest]:
        """Search overlays by name or description"""
        results = []
        
        for overlay in self.overlays.values():
            if industry and overlay.industry != industry:
                continue
            
            if (search_term.lower() in overlay.name.lower() or 
                search_term.lower() in overlay.description.lower()):
                results.append(overlay)
        
        # Sort by trust score
        results.sort(
            key=lambda o: self.overlay_scores.get(o.overlay_id, {}).get('trust_score', 0.0),
            reverse=True
        )
        
        return results
    
    def get_registry_statistics(self) -> Dict[str, Any]:
        """Get registry statistics"""
        total_overlays = len(self.overlays)
        
        overlays_by_industry = {}
        overlays_by_type = {}
        overlays_by_status = {}
        
        for overlay in self.overlays.values():
            # By industry
            industry = overlay.industry.value
            overlays_by_industry[industry] = overlays_by_industry.get(industry, 0) + 1
            
            # By type
            overlay_type = overlay.overlay_type.value
            overlays_by_type[overlay_type] = overlays_by_type.get(overlay_type, 0) + 1
            
            # By status
            status = overlay.status.value
            overlays_by_status[status] = overlays_by_status.get(status, 0) + 1
        
        # Calculate average trust score
        trust_scores = [
            score.get('trust_score', 0.0) 
            for score in self.overlay_scores.values()
        ]
        avg_trust_score = sum(trust_scores) / len(trust_scores) if trust_scores else 0.0
        
        return {
            'total_overlays': total_overlays,
            'overlays_by_industry': overlays_by_industry,
            'overlays_by_type': overlays_by_type,
            'overlays_by_status': overlays_by_status,
            'average_trust_score': round(avg_trust_score, 3),
            'published_overlays': len([o for o in self.overlays.values() if o.status == OverlayStatus.PUBLISHED])
        }

# Global instances
adapter_sdk = AdapterSDK()
overlay_registry = OverlayRegistry()
