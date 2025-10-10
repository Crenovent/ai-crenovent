# SaaS Industry Overlay - Chapter 9.1
# Tasks 9.1-T15: Baseline SaaS overlay (SOX+GDPR; pipeline hygiene)

import json
import uuid
from datetime import datetime, timezone
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict, field
import logging

from ..adapters.adapter_sdk import (
    PolicyAdapter, SchemaAdapter, DSLMacroAdapter, ConnectorAdapter, DashboardAdapter,
    OverlayManifest, IndustryType, OverlayType, OverlayStatus
)

logger = logging.getLogger(__name__)

class SaaSPolicyAdapter:
    """SaaS Policy Adapter - SOX and GDPR compliance policies"""
    
    def validate_policy(self, policy_data: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
        """Validate SaaS policy against SOX and GDPR requirements"""
        validation_result = {
            'valid': True,
            'issues': [],
            'warnings': []
        }
        
        # SOX Revenue Recognition Validation
        if 'revenue_impact' in policy_data:
            revenue_impact = policy_data.get('revenue_impact', 0)
            if revenue_impact > 10000:  # $10K threshold
                if 'cfo_approval' not in policy_data or not policy_data['cfo_approval']:
                    validation_result['issues'].append("Revenue impact >$10K requires CFO approval (SOX compliance)")
                    validation_result['valid'] = False
        
        # GDPR Data Processing Validation
        if 'customer_data_processed' in policy_data and policy_data['customer_data_processed']:
            if 'gdpr_lawful_basis' not in policy_data:
                validation_result['issues'].append("Customer data processing requires GDPR lawful basis")
                validation_result['valid'] = False
            
            if 'data_retention_days' not in policy_data:
                validation_result['warnings'].append("Consider specifying data retention period for GDPR compliance")
        
        # SaaS Subscription Lifecycle Validation
        if 'subscription_changes' in policy_data:
            if 'dual_approval' not in policy_data or not policy_data['dual_approval']:
                validation_result['warnings'].append("Subscription changes should have dual approval for SOX compliance")
        
        return validation_result
    
    def apply_policy_overlay(self, base_policy: Dict[str, Any], overlay: Dict[str, Any]) -> Dict[str, Any]:
        """Apply SaaS policy overlay to base policy"""
        merged_policy = base_policy.copy()
        
        # Add SaaS-specific policy rules
        saas_rules = {
            'sox_revenue_threshold': 10000,
            'gdpr_data_retention_days': 2555,  # 7 years for SOX
            'subscription_approval_required': True,
            'customer_data_encryption': True,
            'audit_trail_required': True
        }
        
        merged_policy.update(saas_rules)
        merged_policy.update(overlay)
        
        return merged_policy

class SaaSSchemaAdapter:
    """SaaS Schema Adapter - Revenue and customer data fields"""
    
    def extend_schema(self, base_schema: Dict[str, Any], extensions: Dict[str, Any]) -> Dict[str, Any]:
        """Extend base schema with SaaS-specific fields"""
        extended_schema = base_schema.copy()
        
        # Add SaaS-specific schema fields
        saas_fields = {
            'properties': {
                **extended_schema.get('properties', {}),
                'arr_impact': {
                    'type': 'number',
                    'description': 'Annual Recurring Revenue impact',
                    'minimum': 0
                },
                'mrr_impact': {
                    'type': 'number',
                    'description': 'Monthly Recurring Revenue impact',
                    'minimum': 0
                },
                'customer_tier': {
                    'type': 'string',
                    'enum': ['enterprise', 'mid_market', 'smb', 'startup'],
                    'description': 'Customer tier classification'
                },
                'subscription_type': {
                    'type': 'string',
                    'enum': ['monthly', 'annual', 'multi_year'],
                    'description': 'Subscription billing type'
                },
                'churn_risk': {
                    'type': 'number',
                    'minimum': 0,
                    'maximum': 1,
                    'description': 'Customer churn risk score'
                },
                'customer_health_score': {
                    'type': 'number',
                    'minimum': 0,
                    'maximum': 100,
                    'description': 'Customer health score'
                },
                'usage_metrics': {
                    'type': 'object',
                    'properties': {
                        'monthly_active_users': {'type': 'integer'},
                        'feature_adoption': {'type': 'number'},
                        'api_calls': {'type': 'integer'}
                    }
                },
                'compliance_flags': {
                    'type': 'object',
                    'properties': {
                        'sox_controlled': {'type': 'boolean'},
                        'gdpr_applicable': {'type': 'boolean'},
                        'pii_processed': {'type': 'boolean'}
                    }
                }
            }
        }
        
        extended_schema.update(saas_fields)
        extended_schema.update(extensions)
        
        return extended_schema
    
    def validate_schema_compatibility(self, base_schema: Dict[str, Any], overlay_schema: Dict[str, Any]) -> bool:
        """Validate SaaS schema compatibility"""
        # Check for required SaaS fields
        required_fields = ['arr_impact', 'customer_tier', 'compliance_flags']
        overlay_properties = overlay_schema.get('properties', {})
        
        for field in required_fields:
            if field not in overlay_properties:
                logger.warning(f"⚠️ SaaS overlay missing recommended field: {field}")
        
        return True

class SaaSDSLMacroAdapter:
    """SaaS DSL Macro Adapter - Pipeline hygiene and revenue workflows"""
    
    def __init__(self):
        self.macros = {
            'pipeline_hygiene_check': {
                'description': 'Check pipeline hygiene for stalled opportunities',
                'parameters': ['stale_days', 'hygiene_threshold'],
                'template': {
                    'type': 'query',
                    'params': {
                        'source': 'salesforce',
                        'query': 'SELECT Id, Name, StageName, LastModifiedDate FROM Opportunity WHERE LastModifiedDate < ${stale_days} DAYS AGO',
                        'filters': {
                            'stage_not_in': ['Closed Won', 'Closed Lost']
                        }
                    }
                }
            },
            'revenue_recognition_check': {
                'description': 'Validate revenue recognition compliance',
                'parameters': ['revenue_threshold', 'approval_required'],
                'template': {
                    'type': 'decision',
                    'params': {
                        'condition': 'revenue_impact > ${revenue_threshold}',
                        'true_action': {
                            'type': 'governance',
                            'params': {
                                'policy_id': 'sox_revenue_approval',
                                'approval_required': '${approval_required}'
                            }
                        }
                    }
                }
            },
            'customer_health_monitor': {
                'description': 'Monitor customer health and churn risk',
                'parameters': ['health_threshold', 'churn_threshold'],
                'template': {
                    'type': 'ml_decision',
                    'params': {
                        'model': 'customer_health_predictor',
                        'features': ['usage_metrics', 'support_tickets', 'payment_history'],
                        'threshold': '${health_threshold}'
                    }
                }
            },
            'gdpr_data_request': {
                'description': 'Handle GDPR data subject requests',
                'parameters': ['request_type', 'customer_id'],
                'template': {
                    'type': 'agent_call',
                    'params': {
                        'agent': 'gdpr_compliance_agent',
                        'action': '${request_type}',
                        'customer_id': '${customer_id}',
                        'compliance_framework': 'GDPR'
                    }
                }
            }
        }
    
    def register_macros(self, macros: Dict[str, Any]) -> bool:
        """Register SaaS-specific DSL macros"""
        try:
            self.macros.update(macros)
            logger.info(f"✅ Registered {len(macros)} SaaS DSL macros")
            return True
        except Exception as e:
            logger.error(f"❌ Failed to register SaaS macros: {e}")
            return False
    
    def expand_macro(self, macro_name: str, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Expand SaaS DSL macro with parameters"""
        if macro_name not in self.macros:
            raise ValueError(f"Unknown SaaS macro: {macro_name}")
        
        macro = self.macros[macro_name]
        template = macro['template'].copy()
        
        # Replace parameter placeholders
        template_str = json.dumps(template)
        for param, value in parameters.items():
            template_str = template_str.replace(f"${{{param}}}", str(value))
        
        return json.loads(template_str)

class SaaSConnectorAdapter:
    """SaaS Connector Adapter - CRM, billing, and customer success integrations"""
    
    def create_connector_shim(self, connector_type: str, config: Dict[str, Any]) -> Dict[str, Any]:
        """Create SaaS-normalized connector shim"""
        
        connector_configs = {
            'salesforce': {
                'type': 'crm',
                'auth_type': 'oauth2',
                'endpoints': {
                    'opportunities': '/services/data/v54.0/sobjects/Opportunity',
                    'accounts': '/services/data/v54.0/sobjects/Account',
                    'contacts': '/services/data/v54.0/sobjects/Contact'
                },
                'rate_limits': {
                    'requests_per_hour': 1000,
                    'burst_limit': 100
                },
                'retry_config': {
                    'max_retries': 3,
                    'backoff_factor': 2
                }
            },
            'stripe': {
                'type': 'billing',
                'auth_type': 'api_key',
                'endpoints': {
                    'subscriptions': '/v1/subscriptions',
                    'customers': '/v1/customers',
                    'invoices': '/v1/invoices'
                },
                'rate_limits': {
                    'requests_per_second': 100
                },
                'retry_config': {
                    'max_retries': 3,
                    'backoff_factor': 1.5
                }
            },
            'hubspot': {
                'type': 'marketing',
                'auth_type': 'oauth2',
                'endpoints': {
                    'contacts': '/crm/v3/objects/contacts',
                    'deals': '/crm/v3/objects/deals',
                    'companies': '/crm/v3/objects/companies'
                },
                'rate_limits': {
                    'requests_per_second': 10
                },
                'retry_config': {
                    'max_retries': 5,
                    'backoff_factor': 2
                }
            }
        }
        
        if connector_type not in connector_configs:
            raise ValueError(f"Unsupported SaaS connector type: {connector_type}")
        
        base_config = connector_configs[connector_type].copy()
        base_config.update(config)
        
        return base_config
    
    def validate_connector_config(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Validate SaaS connector configuration"""
        validation_result = {
            'valid': True,
            'issues': [],
            'warnings': []
        }
        
        required_fields = ['type', 'auth_type', 'endpoints']
        for field in required_fields:
            if field not in config:
                validation_result['issues'].append(f"Missing required field: {field}")
                validation_result['valid'] = False
        
        # Validate rate limits
        if 'rate_limits' not in config:
            validation_result['warnings'].append("No rate limits specified - may cause API throttling")
        
        # Validate retry configuration
        if 'retry_config' not in config:
            validation_result['warnings'].append("No retry configuration - may cause transient failures")
        
        return validation_result

class SaaSDashboardAdapter:
    """SaaS Dashboard Adapter - Revenue, pipeline, and customer success KPIs"""
    
    def __init__(self):
        self.industry_kpis = [
            'arr_growth_rate',
            'mrr_growth_rate',
            'customer_churn_rate',
            'revenue_churn_rate',
            'customer_acquisition_cost',
            'customer_lifetime_value',
            'pipeline_velocity',
            'pipeline_coverage',
            'win_rate',
            'average_deal_size',
            'time_to_close',
            'customer_health_score',
            'product_adoption_rate',
            'support_ticket_volume',
            'nps_score'
        ]
    
    def create_dashboard_config(self, dashboard_type: str, kpis: List[str]) -> Dict[str, Any]:
        """Create SaaS-specific dashboard configuration"""
        
        dashboard_configs = {
            'revenue_dashboard': {
                'title': 'SaaS Revenue Dashboard',
                'description': 'Track ARR, MRR, and revenue growth metrics',
                'layout': 'grid',
                'refresh_interval': 300,  # 5 minutes
                'widgets': [
                    {
                        'type': 'metric_card',
                        'title': 'ARR',
                        'kpi': 'arr_growth_rate',
                        'format': 'currency',
                        'size': 'large'
                    },
                    {
                        'type': 'metric_card',
                        'title': 'MRR',
                        'kpi': 'mrr_growth_rate',
                        'format': 'currency',
                        'size': 'large'
                    },
                    {
                        'type': 'chart',
                        'title': 'Revenue Trend',
                        'chart_type': 'line',
                        'kpis': ['arr_growth_rate', 'mrr_growth_rate'],
                        'time_range': '12M'
                    }
                ]
            },
            'pipeline_dashboard': {
                'title': 'Sales Pipeline Dashboard',
                'description': 'Monitor pipeline health and sales performance',
                'layout': 'grid',
                'refresh_interval': 600,  # 10 minutes
                'widgets': [
                    {
                        'type': 'metric_card',
                        'title': 'Pipeline Coverage',
                        'kpi': 'pipeline_coverage',
                        'format': 'percentage',
                        'size': 'medium'
                    },
                    {
                        'type': 'metric_card',
                        'title': 'Win Rate',
                        'kpi': 'win_rate',
                        'format': 'percentage',
                        'size': 'medium'
                    },
                    {
                        'type': 'funnel_chart',
                        'title': 'Sales Funnel',
                        'stages': ['Lead', 'Qualified', 'Proposal', 'Negotiation', 'Closed Won']
                    }
                ]
            },
            'customer_success_dashboard': {
                'title': 'Customer Success Dashboard',
                'description': 'Track customer health and retention metrics',
                'layout': 'grid',
                'refresh_interval': 900,  # 15 minutes
                'widgets': [
                    {
                        'type': 'metric_card',
                        'title': 'Customer Churn Rate',
                        'kpi': 'customer_churn_rate',
                        'format': 'percentage',
                        'size': 'large'
                    },
                    {
                        'type': 'metric_card',
                        'title': 'NPS Score',
                        'kpi': 'nps_score',
                        'format': 'number',
                        'size': 'medium'
                    },
                    {
                        'type': 'heatmap',
                        'title': 'Customer Health',
                        'kpi': 'customer_health_score',
                        'dimensions': ['customer_tier', 'region']
                    }
                ]
            }
        }
        
        if dashboard_type not in dashboard_configs:
            raise ValueError(f"Unsupported SaaS dashboard type: {dashboard_type}")
        
        config = dashboard_configs[dashboard_type].copy()
        
        # Filter widgets based on requested KPIs
        if kpis:
            filtered_widgets = []
            for widget in config['widgets']:
                widget_kpi = widget.get('kpi')
                widget_kpis = widget.get('kpis', [])
                
                if widget_kpi in kpis or any(k in kpis for k in widget_kpis):
                    filtered_widgets.append(widget)
            
            config['widgets'] = filtered_widgets
        
        return config
    
    def get_industry_kpis(self) -> List[str]:
        """Get SaaS industry-specific KPIs"""
        return self.industry_kpis.copy()

def create_saas_overlay() -> OverlayManifest:
    """Create SaaS industry overlay manifest"""
    
    overlay = OverlayManifest(
        overlay_id="saas_baseline_v1",
        name="SaaS Baseline Overlay",
        version="1.0.0",
        industry=IndustryType.SAAS,
        overlay_type=OverlayType.COMPLIANCE,
        description="Baseline SaaS overlay with SOX and GDPR compliance for pipeline hygiene and revenue recognition",
        created_by="system",
        created_at=datetime.now(timezone.utc).isoformat(),
        compliance_frameworks=["SOX", "GDPR", "CCPA"],
        residency_requirements=["US", "EU"],
        sod_requirements=["maker_checker", "dual_approval"],
        schema_extensions={
            'saas_revenue_fields': {
                'arr_impact': 'number',
                'mrr_impact': 'number',
                'customer_tier': 'string'
            }
        },
        policy_rules={
            'sox_revenue_threshold': 10000,
            'gdpr_data_retention_days': 2555,
            'dual_approval_required': True
        },
        dsl_macros={
            'pipeline_hygiene_check': 'Check pipeline for stalled opportunities',
            'revenue_recognition_check': 'Validate SOX revenue recognition',
            'gdpr_data_request': 'Handle GDPR data subject requests'
        },
        connector_configs={
            'salesforce': 'CRM integration',
            'stripe': 'Billing integration',
            'hubspot': 'Marketing integration'
        },
        status=OverlayStatus.PUBLISHED
    )
    
    return overlay

# Initialize SaaS adapters
saas_policy_adapter = SaaSPolicyAdapter()
saas_schema_adapter = SaaSSchemaAdapter()
saas_dsl_macro_adapter = SaaSDSLMacroAdapter()
saas_connector_adapter = SaaSConnectorAdapter()
saas_dashboard_adapter = SaaSDashboardAdapter()
