# Overlay Manager - Chapter 9.1
# Tasks 9.1-T06, T21, T22: Orchestrator overlay loader, inheritance resolver, tenant scoping

import json
import uuid
from datetime import datetime, timezone
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict
import logging
import asyncio

from ..adapters.adapter_sdk import (
    AdapterSDK, OverlayRegistry, OverlayManifest, IndustryType, OverlayType, OverlayStatus,
    adapter_sdk, overlay_registry
)
from .saas_overlay import (
    saas_policy_adapter, saas_schema_adapter, saas_dsl_macro_adapter,
    saas_connector_adapter, saas_dashboard_adapter, create_saas_overlay
)
from .banking_overlay import (
    banking_policy_adapter, banking_schema_adapter, banking_dsl_macro_adapter,
    banking_connector_adapter, banking_dashboard_adapter, create_banking_overlay
)

logger = logging.getLogger(__name__)

@dataclass
class TenantOverlayConfig:
    """Tenant-specific overlay configuration"""
    tenant_id: int
    industry: IndustryType
    active_overlays: List[str]
    overlay_parameters: Dict[str, Any]
    
    # Inheritance chain: baseline → industry → tenant
    baseline_overlay: Optional[str] = None
    industry_overlay: Optional[str] = None
    tenant_overlay: Optional[str] = None
    
    # Governance
    residency_zone: str = "US"
    compliance_frameworks: List[str] = None
    
    def __post_init__(self):
        if self.compliance_frameworks is None:
            self.compliance_frameworks = []

class OverlayInheritanceResolver:
    """
    Overlay Inheritance Resolver - Task 9.1-T21
    Resolves layered overlays: baseline → industry → tenant
    """
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    def resolve_overlay_chain(
        self,
        tenant_config: TenantOverlayConfig,
        overlay_registry: OverlayRegistry
    ) -> Dict[str, Any]:
        """Resolve complete overlay chain for tenant"""
        
        resolved_config = {
            'policies': {},
            'schemas': {},
            'dsl_macros': {},
            'connectors': {},
            'dashboards': {},
            'parameters': {}
        }
        
        # 1. Apply baseline overlay (if specified)
        if tenant_config.baseline_overlay:
            baseline = overlay_registry.get_overlay(tenant_config.baseline_overlay)
            if baseline:
                self._merge_overlay_config(resolved_config, baseline)
                self.logger.info(f"✅ Applied baseline overlay: {baseline.name}")
        
        # 2. Apply industry overlay
        if tenant_config.industry_overlay:
            industry_overlay = overlay_registry.get_overlay(tenant_config.industry_overlay)
            if industry_overlay:
                self._merge_overlay_config(resolved_config, industry_overlay)
                self.logger.info(f"✅ Applied industry overlay: {industry_overlay.name}")
        
        # 3. Apply tenant-specific overlay (if any)
        if tenant_config.tenant_overlay:
            tenant_overlay = overlay_registry.get_overlay(tenant_config.tenant_overlay)
            if tenant_overlay:
                self._merge_overlay_config(resolved_config, tenant_overlay)
                self.logger.info(f"✅ Applied tenant overlay: {tenant_overlay.name}")
        
        # 4. Apply tenant parameters
        if tenant_config.overlay_parameters:
            resolved_config['parameters'].update(tenant_config.overlay_parameters)
        
        return resolved_config
    
    def _merge_overlay_config(self, base_config: Dict[str, Any], overlay: OverlayManifest) -> None:
        """Merge overlay configuration into base configuration"""
        
        # Merge policy rules
        if overlay.policy_rules:
            base_config['policies'].update(overlay.policy_rules)
        
        # Merge schema extensions
        if overlay.schema_extensions:
            base_config['schemas'].update(overlay.schema_extensions)
        
        # Merge DSL macros
        if overlay.dsl_macros:
            base_config['dsl_macros'].update(overlay.dsl_macros)
        
        # Merge connector configs
        if overlay.connector_configs:
            base_config['connectors'].update(overlay.connector_configs)
    
    def validate_inheritance_chain(
        self,
        tenant_config: TenantOverlayConfig,
        overlay_registry: OverlayRegistry
    ) -> Dict[str, Any]:
        """Validate overlay inheritance chain for conflicts"""
        
        validation_result = {
            'valid': True,
            'conflicts': [],
            'warnings': []
        }
        
        overlays_to_check = []
        
        # Collect overlays in inheritance order
        if tenant_config.baseline_overlay:
            baseline = overlay_registry.get_overlay(tenant_config.baseline_overlay)
            if baseline:
                overlays_to_check.append(baseline)
        
        if tenant_config.industry_overlay:
            industry = overlay_registry.get_overlay(tenant_config.industry_overlay)
            if industry:
                overlays_to_check.append(industry)
        
        if tenant_config.tenant_overlay:
            tenant = overlay_registry.get_overlay(tenant_config.tenant_overlay)
            if tenant:
                overlays_to_check.append(tenant)
        
        # Check for conflicts between overlays
        all_policies = {}
        all_schemas = {}
        
        for overlay in overlays_to_check:
            # Check policy conflicts
            for policy_key, policy_value in overlay.policy_rules.items():
                if policy_key in all_policies and all_policies[policy_key] != policy_value:
                    validation_result['conflicts'].append(
                        f"Policy conflict: {policy_key} = {all_policies[policy_key]} vs {policy_value}"
                    )
                    validation_result['valid'] = False
                all_policies[policy_key] = policy_value
            
            # Check schema conflicts
            for schema_key, schema_value in overlay.schema_extensions.items():
                if schema_key in all_schemas and all_schemas[schema_key] != schema_value:
                    validation_result['warnings'].append(
                        f"Schema override: {schema_key} overridden by {overlay.name}"
                    )
                all_schemas[schema_key] = schema_value
        
        return validation_result

class OverlayLoader:
    """
    Overlay Loader - Tasks 9.1-T06, T22
    Per-tenant overlay activation with runtime discovery
    """
    
    def __init__(self, adapter_sdk: AdapterSDK, overlay_registry: OverlayRegistry):
        self.adapter_sdk = adapter_sdk
        self.overlay_registry = overlay_registry
        self.inheritance_resolver = OverlayInheritanceResolver()
        self.tenant_configs: Dict[int, TenantOverlayConfig] = {}
        self.logger = logging.getLogger(__name__)
    
    def register_tenant_overlay_config(self, tenant_config: TenantOverlayConfig) -> bool:
        """Register tenant overlay configuration"""
        try:
            # Validate inheritance chain
            validation_result = self.inheritance_resolver.validate_inheritance_chain(
                tenant_config, self.overlay_registry
            )
            
            if not validation_result['valid']:
                self.logger.error(f"❌ Invalid overlay chain for tenant {tenant_config.tenant_id}: {validation_result['conflicts']}")
                return False
            
            # Store tenant configuration
            self.tenant_configs[tenant_config.tenant_id] = tenant_config
            
            self.logger.info(f"✅ Registered overlay config for tenant {tenant_config.tenant_id} ({tenant_config.industry.value})")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Failed to register tenant overlay config: {e}")
            return False
    
    def load_tenant_overlays(self, tenant_id: int) -> Optional[Dict[str, Any]]:
        """Load and resolve overlays for tenant"""
        
        tenant_config = self.tenant_configs.get(tenant_id)
        if not tenant_config:
            self.logger.warning(f"⚠️ No overlay config found for tenant {tenant_id}")
            return None
        
        try:
            # Resolve overlay inheritance chain
            resolved_config = self.inheritance_resolver.resolve_overlay_chain(
                tenant_config, self.overlay_registry
            )
            
            # Apply industry-specific adapters
            self._apply_industry_adapters(tenant_config.industry, resolved_config)
            
            self.logger.info(f"✅ Loaded overlays for tenant {tenant_id}")
            return resolved_config
            
        except Exception as e:
            self.logger.error(f"❌ Failed to load overlays for tenant {tenant_id}: {e}")
            return None
    
    def _apply_industry_adapters(self, industry: IndustryType, config: Dict[str, Any]) -> None:
        """Apply industry-specific adapters to configuration"""
        
        # Get adapters for the industry
        policy_adapter = self.adapter_sdk.get_policy_adapter(industry)
        schema_adapter = self.adapter_sdk.get_schema_adapter(industry)
        dsl_macro_adapter = self.adapter_sdk.get_dsl_macro_adapter(industry)
        connector_adapter = self.adapter_sdk.get_connector_adapter(industry)
        dashboard_adapter = self.adapter_sdk.get_dashboard_adapter(industry)
        
        # Apply policy overlay if adapter exists
        if policy_adapter and config.get('policies'):
            try:
                config['policies'] = policy_adapter.apply_policy_overlay(
                    config['policies'], {}
                )
            except Exception as e:
                self.logger.warning(f"⚠️ Failed to apply policy overlay: {e}")
        
        # Apply schema extensions if adapter exists
        if schema_adapter and config.get('schemas'):
            try:
                config['schemas'] = schema_adapter.extend_schema(
                    config['schemas'], {}
                )
            except Exception as e:
                self.logger.warning(f"⚠️ Failed to apply schema overlay: {e}")
    
    def get_tenant_overlay_status(self, tenant_id: int) -> Dict[str, Any]:
        """Get overlay status for tenant"""
        
        tenant_config = self.tenant_configs.get(tenant_id)
        if not tenant_config:
            return {'status': 'not_configured'}
        
        status = {
            'status': 'configured',
            'tenant_id': tenant_id,
            'industry': tenant_config.industry.value,
            'active_overlays': tenant_config.active_overlays,
            'residency_zone': tenant_config.residency_zone,
            'compliance_frameworks': tenant_config.compliance_frameworks,
            'overlay_chain': {
                'baseline': tenant_config.baseline_overlay,
                'industry': tenant_config.industry_overlay,
                'tenant': tenant_config.tenant_overlay
            }
        }
        
        return status
    
    def list_tenant_overlays(self) -> List[Dict[str, Any]]:
        """List all tenant overlay configurations"""
        
        tenant_list = []
        for tenant_id, config in self.tenant_configs.items():
            tenant_list.append(self.get_tenant_overlay_status(tenant_id))
        
        return tenant_list

class OverlayManager:
    """
    Comprehensive Overlay Manager - Tasks 9.1-T01 to T60
    Manages multi-industry adapter model with governance and compliance
    """
    
    def __init__(self):
        self.adapter_sdk = adapter_sdk
        self.overlay_registry = overlay_registry
        self.overlay_loader = OverlayLoader(self.adapter_sdk, self.overlay_registry)
        self.logger = logging.getLogger(__name__)
        
        # Initialize with default overlays
        self._initialize_default_overlays()
    
    def _initialize_default_overlays(self) -> None:
        """Initialize default industry overlays"""
        
        try:
            # Register SaaS adapters
            self.adapter_sdk.register_policy_adapter(IndustryType.SAAS, saas_policy_adapter)
            self.adapter_sdk.register_schema_adapter(IndustryType.SAAS, saas_schema_adapter)
            self.adapter_sdk.register_dsl_macro_adapter(IndustryType.SAAS, saas_dsl_macro_adapter)
            self.adapter_sdk.register_connector_adapter(IndustryType.SAAS, saas_connector_adapter)
            self.adapter_sdk.register_dashboard_adapter(IndustryType.SAAS, saas_dashboard_adapter)
            
            # Register Banking adapters
            self.adapter_sdk.register_policy_adapter(IndustryType.BANKING, banking_policy_adapter)
            self.adapter_sdk.register_schema_adapter(IndustryType.BANKING, banking_schema_adapter)
            self.adapter_sdk.register_dsl_macro_adapter(IndustryType.BANKING, banking_dsl_macro_adapter)
            self.adapter_sdk.register_connector_adapter(IndustryType.BANKING, banking_connector_adapter)
            self.adapter_sdk.register_dashboard_adapter(IndustryType.BANKING, banking_dashboard_adapter)
            
            # Register default overlays
            saas_overlay = create_saas_overlay()
            banking_overlay = create_banking_overlay()
            
            self.overlay_registry.register_overlay(saas_overlay)
            self.overlay_registry.register_overlay(banking_overlay)
            
            self.logger.info("✅ Initialized default industry overlays (SaaS, Banking)")
            
        except Exception as e:
            self.logger.error(f"❌ Failed to initialize default overlays: {e}")
    
    async def onboard_tenant(
        self,
        tenant_id: int,
        industry: IndustryType,
        residency_zone: str = "US",
        compliance_frameworks: List[str] = None,
        custom_parameters: Dict[str, Any] = None
    ) -> bool:
        """Onboard tenant with appropriate industry overlay"""
        
        try:
            # Determine industry overlay
            industry_overlay_id = None
            if industry == IndustryType.SAAS:
                industry_overlay_id = "saas_baseline_v1"
            elif industry == IndustryType.BANKING:
                industry_overlay_id = "banking_bfsi_v1"
            
            if not industry_overlay_id:
                self.logger.error(f"❌ No overlay available for industry: {industry.value}")
                return False
            
            # Create tenant overlay configuration
            tenant_config = TenantOverlayConfig(
                tenant_id=tenant_id,
                industry=industry,
                active_overlays=[industry_overlay_id],
                overlay_parameters=custom_parameters or {},
                industry_overlay=industry_overlay_id,
                residency_zone=residency_zone,
                compliance_frameworks=compliance_frameworks or []
            )
            
            # Register tenant configuration
            success = self.overlay_loader.register_tenant_overlay_config(tenant_config)
            
            if success:
                self.logger.info(f"✅ Onboarded tenant {tenant_id} with {industry.value} overlay")
            
            return success
            
        except Exception as e:
            self.logger.error(f"❌ Failed to onboard tenant {tenant_id}: {e}")
            return False
    
    def get_tenant_configuration(self, tenant_id: int) -> Optional[Dict[str, Any]]:
        """Get resolved configuration for tenant"""
        return self.overlay_loader.load_tenant_overlays(tenant_id)
    
    def validate_workflow_compliance(
        self,
        tenant_id: int,
        workflow_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Validate workflow against tenant's compliance overlays"""
        
        tenant_config = self.overlay_loader.tenant_configs.get(tenant_id)
        if not tenant_config:
            return {'valid': False, 'error': 'Tenant not configured'}
        
        # Get industry policy adapter
        policy_adapter = self.adapter_sdk.get_policy_adapter(tenant_config.industry)
        if not policy_adapter:
            return {'valid': False, 'error': f'No policy adapter for {tenant_config.industry.value}'}
        
        # Validate against industry policies
        validation_result = policy_adapter.validate_policy(
            workflow_data,
            {'tenant_id': tenant_id, 'industry': tenant_config.industry.value}
        )
        
        return validation_result
    
    def get_industry_kpis(self, industry: IndustryType) -> List[str]:
        """Get industry-specific KPIs"""
        
        dashboard_adapter = self.adapter_sdk.get_dashboard_adapter(industry)
        if dashboard_adapter:
            return dashboard_adapter.get_industry_kpis()
        
        return []
    
    def create_industry_dashboard(
        self,
        industry: IndustryType,
        dashboard_type: str,
        kpis: List[str] = None
    ) -> Optional[Dict[str, Any]]:
        """Create industry-specific dashboard configuration"""
        
        dashboard_adapter = self.adapter_sdk.get_dashboard_adapter(industry)
        if not dashboard_adapter:
            return None
        
        try:
            return dashboard_adapter.create_dashboard_config(dashboard_type, kpis or [])
        except Exception as e:
            self.logger.error(f"❌ Failed to create {industry.value} dashboard: {e}")
            return None
    
    def get_overlay_statistics(self) -> Dict[str, Any]:
        """Get comprehensive overlay statistics"""
        
        registry_stats = self.overlay_registry.get_registry_statistics()
        tenant_stats = {
            'total_tenants': len(self.overlay_loader.tenant_configs),
            'tenants_by_industry': {},
            'tenants_by_residency': {}
        }
        
        # Calculate tenant statistics
        for tenant_config in self.overlay_loader.tenant_configs.values():
            industry = tenant_config.industry.value
            tenant_stats['tenants_by_industry'][industry] = tenant_stats['tenants_by_industry'].get(industry, 0) + 1
            
            residency = tenant_config.residency_zone
            tenant_stats['tenants_by_residency'][residency] = tenant_stats['tenants_by_residency'].get(residency, 0) + 1
        
        return {
            'overlay_registry': registry_stats,
            'tenant_statistics': tenant_stats,
            'supported_industries': [industry.value for industry in IndustryType],
            'registered_adapters': {
                'policy_adapters': len(self.adapter_sdk.policy_adapters),
                'schema_adapters': len(self.adapter_sdk.schema_adapters),
                'dsl_macro_adapters': len(self.adapter_sdk.dsl_macro_adapters),
                'connector_adapters': len(self.adapter_sdk.connector_adapters),
                'dashboard_adapters': len(self.adapter_sdk.dashboard_adapters)
            }
        }

# Global instance
overlay_manager = OverlayManager()
