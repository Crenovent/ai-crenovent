"""
DSL Primitive Library - Task 7.1.11
===================================

Build DSL library of primitives - Standard reusable functions
YAML/JSON templates for connectors, agents, workflows

Following user rules:
- No hardcoding, dynamic and adaptive
- Configuration-driven architecture
- SaaS/IT industry focus
- Multi-tenant aware
- Modular design
"""

import yaml
import json
import logging
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field, asdict
from enum import Enum
from pathlib import Path

logger = logging.getLogger(__name__)

class PrimitiveCategory(Enum):
    """Categories of DSL primitives"""
    CONNECTOR = "connector"
    WORKFLOW = "workflow" 
    AGENT = "agent"
    POLICY = "policy"
    DASHBOARD = "dashboard"
    TEMPLATE = "template"

class IndustryOverlay(Enum):
    """Industry-specific overlays - SaaS/IT focus per user rules"""
    SAAS = "saas"
    IT_SERVICES = "it_services"
    FINTECH = "fintech"
    ECOMMERCE = "ecommerce"
    GENERAL = "general"

@dataclass
class DSLPrimitive:
    """Standard DSL primitive definition"""
    primitive_id: str
    name: str
    description: str
    category: PrimitiveCategory
    industry_overlay: IndustryOverlay
    version: str = "1.0.0"
    
    # Template definition
    template: Dict[str, Any] = field(default_factory=dict)
    parameters: Dict[str, Any] = field(default_factory=dict)
    governance: Dict[str, Any] = field(default_factory=dict)
    
    # Metadata
    tags: List[str] = field(default_factory=list)
    author: str = "system"
    tenant_scoped: bool = True
    
    # Usage tracking
    usage_count: int = 0
    trust_score: float = 0.8
    
class DSLPrimitiveLibrary:
    """
    Comprehensive DSL Primitive Library
    
    Implements Task 7.1.11: Build DSL library of primitives
    - Standard reusable functions
    - YAML/JSON templates  
    - Connectors, agents, workflows
    - Industry-specific overlays (SaaS/IT focus)
    """
    
    def __init__(self, config_path: Optional[str] = None):
        self.logger = logging.getLogger(__name__)
        self.primitives: Dict[str, DSLPrimitive] = {}
        self.config_path = config_path
        
        # Load configuration dynamically (following user rules)
        self._load_dynamic_config()
        
        # Initialize primitive library
        self._initialize_primitive_library()
    
    def _load_dynamic_config(self):
        """Load dynamic configuration from universal parameters"""
        try:
            if self.config_path:
                config_file = Path(self.config_path)
            else:
                config_file = Path(__file__).parent.parent / "configuration" / "universal_rba_parameters.yaml"
            
            if config_file.exists():
                with open(config_file, 'r') as f:
                    self.config = yaml.safe_load(f)
                    self.logger.info("✅ Loaded dynamic primitive library configuration")
            else:
                self.config = {}
                self.logger.warning("No configuration file found, using defaults")
        except Exception as e:
            self.logger.warning(f"Could not load config: {e}, using defaults")
            self.config = {}
    
    def _initialize_primitive_library(self):
        """Initialize the primitive library with standard templates"""
        
        # SaaS-focused connectors (per user rules)
        self._add_saas_connectors()
        
        # IT Services connectors
        self._add_it_services_connectors()
        
        # Standard workflow templates
        self._add_workflow_templates()
        
        # AI Agent templates
        self._add_agent_templates()
        
        # Policy templates
        self._add_policy_templates()
        
        self.logger.info(f"✅ Initialized primitive library with {len(self.primitives)} primitives")
    
    def _add_saas_connectors(self):
        """Add SaaS-specific connector primitives"""
        
        # Salesforce CRM Connector
        salesforce_connector = DSLPrimitive(
            primitive_id="salesforce_crm_connector",
            name="Salesforce CRM Connector",
            description="Connect to Salesforce for opportunity, account, and lead data",
            category=PrimitiveCategory.CONNECTOR,
            industry_overlay=IndustryOverlay.SAAS,
            template={
                "type": "query",
                "params": {
                    "source": "salesforce",
                    "resource": "{{resource_type}}",  # Dynamic parameter
                    "filters": "{{filters}}",
                    "select": "{{fields}}",
                    "limit": "{{limit|default(1000)}}"
                },
                "governance": {
                    "policy_id": "{{policy_pack_id}}",
                    "evidence_capture": True,
                    "consent_required": True,
                    "residency_aware": True
                }
            },
            parameters={
                "resource_type": {"type": "string", "required": True, "options": ["Opportunity", "Account", "Lead", "Contact"]},
                "filters": {"type": "array", "required": False},
                "fields": {"type": "array", "required": False},
                "limit": {"type": "integer", "default": 1000}
            },
            tags=["crm", "salesforce", "saas", "revenue"]
        )
        self.primitives[salesforce_connector.primitive_id] = salesforce_connector
        
        # HubSpot Connector
        hubspot_connector = DSLPrimitive(
            primitive_id="hubspot_crm_connector",
            name="HubSpot CRM Connector", 
            description="Connect to HubSpot for marketing and sales data",
            category=PrimitiveCategory.CONNECTOR,
            industry_overlay=IndustryOverlay.SAAS,
            template={
                "type": "query",
                "params": {
                    "source": "hubspot",
                    "resource": "{{resource_type}}",
                    "filters": "{{filters}}",
                    "properties": "{{properties}}"
                },
                "governance": {
                    "policy_id": "{{policy_pack_id}}",
                    "evidence_capture": True,
                    "gdpr_compliant": True
                }
            },
            parameters={
                "resource_type": {"type": "string", "required": True, "options": ["deals", "contacts", "companies", "tickets"]},
                "filters": {"type": "object", "required": False},
                "properties": {"type": "array", "required": False}
            },
            tags=["crm", "hubspot", "saas", "marketing"]
        )
        self.primitives[hubspot_connector.primitive_id] = hubspot_connector
        
        # Stripe Billing Connector
        stripe_connector = DSLPrimitive(
            primitive_id="stripe_billing_connector",
            name="Stripe Billing Connector",
            description="Connect to Stripe for subscription and payment data",
            category=PrimitiveCategory.CONNECTOR,
            industry_overlay=IndustryOverlay.SAAS,
            template={
                "type": "query",
                "params": {
                    "source": "stripe",
                    "resource": "{{resource_type}}",
                    "filters": "{{filters}}",
                    "expand": "{{expand_fields}}"
                },
                "governance": {
                    "policy_id": "{{policy_pack_id}}",
                    "evidence_capture": True,
                    "pci_compliant": True,
                    "financial_data": True
                }
            },
            parameters={
                "resource_type": {"type": "string", "required": True, "options": ["subscriptions", "customers", "invoices", "charges"]},
                "filters": {"type": "object", "required": False},
                "expand_fields": {"type": "array", "required": False}
            },
            tags=["billing", "stripe", "saas", "payments", "subscription"]
        )
        self.primitives[stripe_connector.primitive_id] = stripe_connector
    
    def _add_it_services_connectors(self):
        """Add IT Services connector primitives"""
        
        # Jira Service Management
        jira_connector = DSLPrimitive(
            primitive_id="jira_service_connector",
            name="Jira Service Management Connector",
            description="Connect to Jira for ticket and project data",
            category=PrimitiveCategory.CONNECTOR,
            industry_overlay=IndustryOverlay.IT_SERVICES,
            template={
                "type": "query",
                "params": {
                    "source": "jira",
                    "resource": "{{resource_type}}",
                    "jql": "{{jql_query}}",
                    "fields": "{{fields}}"
                },
                "governance": {
                    "policy_id": "{{policy_pack_id}}",
                    "evidence_capture": True,
                    "service_data": True
                }
            },
            parameters={
                "resource_type": {"type": "string", "required": True, "options": ["issues", "projects", "users"]},
                "jql_query": {"type": "string", "required": False},
                "fields": {"type": "array", "required": False}
            },
            tags=["itsm", "jira", "tickets", "projects"]
        )
        self.primitives[jira_connector.primitive_id] = jira_connector
        
        # ServiceNow Connector
        servicenow_connector = DSLPrimitive(
            primitive_id="servicenow_itsm_connector",
            name="ServiceNow ITSM Connector",
            description="Connect to ServiceNow for IT service management data",
            category=PrimitiveCategory.CONNECTOR,
            industry_overlay=IndustryOverlay.IT_SERVICES,
            template={
                "type": "query",
                "params": {
                    "source": "servicenow",
                    "table": "{{table_name}}",
                    "query": "{{query_filter}}",
                    "fields": "{{fields}}"
                },
                "governance": {
                    "policy_id": "{{policy_pack_id}}",
                    "evidence_capture": True,
                    "itsm_compliant": True
                }
            },
            parameters={
                "table_name": {"type": "string", "required": True, "options": ["incident", "change_request", "problem", "task"]},
                "query_filter": {"type": "string", "required": False},
                "fields": {"type": "array", "required": False}
            },
            tags=["itsm", "servicenow", "incidents", "changes"]
        )
        self.primitives[servicenow_connector.primitive_id] = servicenow_connector
    
    def _add_workflow_templates(self):
        """Add standard workflow templates"""
        
        # SaaS Pipeline Hygiene Workflow
        pipeline_hygiene = DSLPrimitive(
            primitive_id="saas_pipeline_hygiene_workflow",
            name="SaaS Pipeline Hygiene Workflow",
            description="Automated pipeline hygiene checks for SaaS revenue operations",
            category=PrimitiveCategory.WORKFLOW,
            industry_overlay=IndustryOverlay.SAAS,
            template={
                "workflow_id": "{{workflow_id}}",
                "name": "{{workflow_name|default('Pipeline Hygiene Check')}}",
                "version": "1.0.0",
                "governance": {
                    "policy_packs": ["{{policy_pack_id}}", "sox_compliance"],
                    "evidence_required": True,
                    "tenant_id": "{{tenant_id}}"
                },
                "steps": [
                    {
                        "id": "fetch_stale_opportunities",
                        "type": "query",
                        "params": {
                            "source": "salesforce",
                            "resource": "Opportunity",
                            "filters": [
                                {"field": "LastModifiedDate", "op": "LESS_THAN", "value": "{{stale_days|default(30)}} DAYS AGO"},
                                {"field": "IsClosed", "op": "EQUALS", "value": False}
                            ]
                        }
                    },
                    {
                        "id": "check_hygiene_rules",
                        "type": "decision",
                        "params": {
                            "expression": "{{stale_opportunities.count}} > {{threshold|default(10)}}"
                        },
                        "on_true": "notify_sales_ops",
                        "on_false": "log_success"
                    },
                    {
                        "id": "notify_sales_ops",
                        "type": "notify",
                        "params": {
                            "channel": "slack",
                            "to": ["{{sales_ops_channel|default('#sales-ops')}}"],
                            "template": "Found {{stale_opportunities.count}} stale opportunities requiring attention"
                        }
                    }
                ]
            },
            parameters={
                "workflow_id": {"type": "string", "required": True},
                "workflow_name": {"type": "string", "default": "Pipeline Hygiene Check"},
                "stale_days": {"type": "integer", "default": 30},
                "threshold": {"type": "integer", "default": 10},
                "sales_ops_channel": {"type": "string", "default": "#sales-ops"}
            },
            tags=["pipeline", "hygiene", "saas", "automation", "revenue"]
        )
        self.primitives[pipeline_hygiene.primitive_id] = pipeline_hygiene
        
        # Customer Success Health Check
        customer_health = DSLPrimitive(
            primitive_id="customer_success_health_check",
            name="Customer Success Health Check",
            description="Monitor customer health metrics and trigger interventions",
            category=PrimitiveCategory.WORKFLOW,
            industry_overlay=IndustryOverlay.SAAS,
            template={
                "workflow_id": "{{workflow_id}}",
                "name": "Customer Health Monitoring",
                "steps": [
                    {
                        "id": "fetch_customer_metrics",
                        "type": "query",
                        "params": {
                            "source": "{{data_source|default('salesforce')}}",
                            "resource": "Account",
                            "filters": [
                                {"field": "Type", "op": "EQUALS", "value": "Customer"}
                            ]
                        }
                    },
                    {
                        "id": "calculate_health_score",
                        "type": "ml_decision",
                        "params": {
                            "model_id": "customer_health_model_v1",
                            "inputs": {
                                "usage_metrics": "{{customer_metrics.usage}}",
                                "support_tickets": "{{customer_metrics.tickets}}",
                                "payment_history": "{{customer_metrics.payments}}"
                            },
                            "thresholds": {
                                "healthy": 0.7,
                                "at_risk": 0.4
                            }
                        }
                    },
                    {
                        "id": "trigger_intervention",
                        "type": "agent_call",
                        "params": {
                            "agent_id": "customer_success_agent",
                            "context": {"health_scores": "{{health_results}}"},
                            "tools": ["email_tool", "task_creator"]
                        }
                    }
                ]
            },
            parameters={
                "workflow_id": {"type": "string", "required": True},
                "data_source": {"type": "string", "default": "salesforce"}
            },
            tags=["customer_success", "health_score", "saas", "ml", "intervention"]
        )
        self.primitives[customer_health.primitive_id] = customer_health
    
    def _add_agent_templates(self):
        """Add AI agent templates"""
        
        # Revenue Forecast Agent
        forecast_agent = DSLPrimitive(
            primitive_id="revenue_forecast_agent",
            name="Revenue Forecast Agent",
            description="AI agent for revenue forecasting and variance analysis",
            category=PrimitiveCategory.AGENT,
            industry_overlay=IndustryOverlay.SAAS,
            template={
                "agent_id": "{{agent_id}}",
                "name": "Revenue Forecast Agent",
                "type": "agent_call",
                "params": {
                    "agent_id": "{{agent_id}}",
                    "context": {
                        "opportunities": "{{opportunity_data}}",
                        "historical_data": "{{historical_data}}",
                        "forecast_period": "{{forecast_period|default('quarterly')}}"
                    },
                    "tools": ["sql_tool", "analytics_tool", "notification_tool"],
                    "max_depth": 5,
                    "timeout_sec": 300,
                    "confidence_floor": 0.7
                },
                "governance": {
                    "policy_id": "{{policy_pack_id}}",
                    "evidence_capture": True,
                    "trust_threshold": 0.8,
                    "override_ledger_id": "{{override_ledger_id}}"
                }
            },
            parameters={
                "agent_id": {"type": "string", "required": True},
                "opportunity_data": {"type": "object", "required": True},
                "historical_data": {"type": "object", "required": False},
                "forecast_period": {"type": "string", "default": "quarterly", "options": ["monthly", "quarterly", "annual"]}
            },
            tags=["forecast", "revenue", "ai_agent", "saas", "analytics"]
        )
        self.primitives[forecast_agent.primitive_id] = forecast_agent
        
        # Customer Churn Prevention Agent
        churn_agent = DSLPrimitive(
            primitive_id="churn_prevention_agent",
            name="Customer Churn Prevention Agent",
            description="AI agent for identifying and preventing customer churn",
            category=PrimitiveCategory.AGENT,
            industry_overlay=IndustryOverlay.SAAS,
            template={
                "agent_id": "{{agent_id}}",
                "name": "Churn Prevention Agent",
                "type": "agent_call",
                "params": {
                    "agent_id": "{{agent_id}}",
                    "context": {
                        "customer_data": "{{customer_data}}",
                        "usage_metrics": "{{usage_metrics}}",
                        "support_history": "{{support_history}}"
                    },
                    "tools": ["ml_model_tool", "email_tool", "task_creator", "slack_notifier"],
                    "max_depth": 4,
                    "timeout_sec": 180
                },
                "governance": {
                    "policy_id": "{{policy_pack_id}}",
                    "evidence_capture": True,
                    "customer_data_protection": True
                }
            },
            parameters={
                "agent_id": {"type": "string", "required": True},
                "customer_data": {"type": "object", "required": True},
                "usage_metrics": {"type": "object", "required": False},
                "support_history": {"type": "object", "required": False}
            },
            tags=["churn", "prevention", "ai_agent", "saas", "customer_success"]
        )
        self.primitives[churn_agent.primitive_id] = churn_agent
    
    def _add_policy_templates(self):
        """Add governance policy templates"""
        
        # SaaS Data Governance Policy
        saas_governance = DSLPrimitive(
            primitive_id="saas_data_governance_policy",
            name="SaaS Data Governance Policy",
            description="Standard data governance policy for SaaS companies",
            category=PrimitiveCategory.POLICY,
            industry_overlay=IndustryOverlay.SAAS,
            template={
                "policy_id": "{{policy_id}}",
                "name": "SaaS Data Governance",
                "type": "governance",
                "params": {
                    "action": "assert",
                    "policy_id": "{{policy_id}}",
                    "fields": {
                        "tenant_id": "{{tenant_id}}",
                        "region": "{{region}}",
                        "data_classification": "{{data_classification|default('internal')}}",
                        "retention_period": "{{retention_days|default(2555)}}"
                    }
                },
                "rules": [
                    {
                        "name": "customer_data_protection",
                        "condition": "data_type == 'customer'",
                        "requirements": ["consent_required", "encryption_at_rest", "audit_logging"]
                    },
                    {
                        "name": "financial_data_protection", 
                        "condition": "data_type == 'financial'",
                        "requirements": ["sox_compliance", "segregation_of_duties", "immutable_audit"]
                    }
                ]
            },
            parameters={
                "policy_id": {"type": "string", "required": True},
                "tenant_id": {"type": "integer", "required": True},
                "region": {"type": "string", "required": True},
                "data_classification": {"type": "string", "default": "internal"},
                "retention_days": {"type": "integer", "default": 2555}
            },
            tags=["governance", "policy", "saas", "data_protection", "compliance"]
        )
        self.primitives[saas_governance.primitive_id] = saas_governance
    
    def get_primitive(self, primitive_id: str) -> Optional[DSLPrimitive]:
        """Get a primitive by ID"""
        return self.primitives.get(primitive_id)
    
    def list_primitives(self, 
                       category: Optional[PrimitiveCategory] = None,
                       industry: Optional[IndustryOverlay] = None,
                       tags: Optional[List[str]] = None) -> List[DSLPrimitive]:
        """List primitives with optional filtering"""
        results = list(self.primitives.values())
        
        if category:
            results = [p for p in results if p.category == category]
        
        if industry:
            results = [p for p in results if p.industry_overlay == industry]
        
        if tags:
            results = [p for p in results if any(tag in p.tags for tag in tags)]
        
        return results
    
    def render_primitive(self, primitive_id: str, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Render a primitive template with parameters"""
        primitive = self.get_primitive(primitive_id)
        if not primitive:
            raise ValueError(f"Primitive {primitive_id} not found")
        
        # Simple template rendering (could be enhanced with Jinja2)
        template_str = json.dumps(primitive.template)
        
        # Replace template variables
        for key, value in parameters.items():
            template_str = template_str.replace(f"{{{{{key}}}}}", str(value))
        
        # Handle default values
        import re
        default_pattern = r'\{\{(\w+)\|default\(([^)]+)\)\}\}'
        def replace_defaults(match):
            var_name, default_value = match.groups()
            if var_name in parameters:
                return str(parameters[var_name])
            else:
                return default_value.strip('"\'')
        
        template_str = re.sub(default_pattern, replace_defaults, template_str)
        
        return json.loads(template_str)
    
    def export_library(self, format: str = "yaml") -> str:
        """Export the entire primitive library"""
        library_data = {
            "primitive_library": {
                "version": "1.0.0",
                "primitives": {pid: asdict(primitive) for pid, primitive in self.primitives.items()}
            }
        }
        
        if format.lower() == "yaml":
            return yaml.dump(library_data, default_flow_style=False)
        elif format.lower() == "json":
            return json.dumps(library_data, indent=2)
        else:
            raise ValueError(f"Unsupported format: {format}")
    
    def add_custom_primitive(self, primitive: DSLPrimitive):
        """Add a custom primitive to the library"""
        self.primitives[primitive.primitive_id] = primitive
        self.logger.info(f"✅ Added custom primitive: {primitive.primitive_id}")
    
    def get_usage_stats(self) -> Dict[str, Any]:
        """Get usage statistics for the primitive library"""
        stats = {
            "total_primitives": len(self.primitives),
            "by_category": {},
            "by_industry": {},
            "most_used": [],
            "highest_trust": []
        }
        
        # Category breakdown
        for primitive in self.primitives.values():
            category = primitive.category.value
            stats["by_category"][category] = stats["by_category"].get(category, 0) + 1
            
            industry = primitive.industry_overlay.value
            stats["by_industry"][industry] = stats["by_industry"].get(industry, 0) + 1
        
        # Most used primitives
        sorted_by_usage = sorted(self.primitives.values(), key=lambda p: p.usage_count, reverse=True)
        stats["most_used"] = [(p.primitive_id, p.usage_count) for p in sorted_by_usage[:5]]
        
        # Highest trust primitives
        sorted_by_trust = sorted(self.primitives.values(), key=lambda p: p.trust_score, reverse=True)
        stats["highest_trust"] = [(p.primitive_id, p.trust_score) for p in sorted_by_trust[:5]]
        
        return stats


# Global instance for easy access
primitive_library = DSLPrimitiveLibrary()

def get_primitive_library() -> DSLPrimitiveLibrary:
    """Get the global primitive library instance"""
    return primitive_library
