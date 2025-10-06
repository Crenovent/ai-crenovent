"""
Persona-Specific DSL Libraries - Task 7.1.18
============================================

Create persona-specific DSL libraries for different user roles:
- CRO (Chief Revenue Officer)
- Sales Manager
- Revenue Operations
- Customer Success Manager
- Finance/Accounting

Following user rules:
- No hardcoding, dynamic configuration
- SaaS/IT industry focus
- Multi-tenant aware
- Modular design
"""

import yaml
import json
import logging
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field
from enum import Enum
from .primitive_library import DSLPrimitive, PrimitiveCategory, IndustryOverlay, DSLPrimitiveLibrary

logger = logging.getLogger(__name__)

class PersonaType(Enum):
    """User persona types"""
    CRO = "cro"
    SALES_MANAGER = "sales_manager"
    REVENUE_OPS = "revenue_ops"
    CUSTOMER_SUCCESS = "customer_success"
    FINANCE = "finance"
    ADMIN = "admin"

@dataclass
class PersonaLibrary:
    """Persona-specific DSL library"""
    persona: PersonaType
    name: str
    description: str
    primitives: List[str] = field(default_factory=list)  # Primitive IDs
    workflows: List[str] = field(default_factory=list)   # Workflow template IDs
    permissions: List[str] = field(default_factory=list) # Required permissions
    industry_focus: IndustryOverlay = IndustryOverlay.SAAS

class PersonaSpecificLibraries:
    """
    Persona-Specific DSL Libraries Manager
    
    Implements Task 7.1.18: Create persona-specific DSL libraries
    - Role-based primitive collections
    - Workflow templates per persona
    - Permission-aware access
    - Industry-specific customization (SaaS/IT focus)
    """
    
    def __init__(self, base_library: DSLPrimitiveLibrary):
        self.logger = logging.getLogger(__name__)
        self.base_library = base_library
        self.persona_libraries: Dict[PersonaType, PersonaLibrary] = {}
        
        # Initialize persona-specific libraries
        self._initialize_persona_libraries()
    
    def _initialize_persona_libraries(self):
        """Initialize persona-specific libraries"""
        
        # CRO Library - Executive-level revenue insights
        self._create_cro_library()
        
        # Sales Manager Library - Team management and pipeline oversight
        self._create_sales_manager_library()
        
        # Revenue Operations Library - Process automation and analytics
        self._create_revenue_ops_library()
        
        # Customer Success Library - Customer health and retention
        self._create_customer_success_library()
        
        # Finance Library - Revenue recognition and compliance
        self._create_finance_library()
        
        self.logger.info(f"✅ Initialized {len(self.persona_libraries)} persona-specific libraries")
    
    def _create_cro_library(self):
        """Create CRO-specific DSL library"""
        cro_library = PersonaLibrary(
            persona=PersonaType.CRO,
            name="Chief Revenue Officer Library",
            description="Executive-level revenue insights, forecasting, and strategic analysis",
            permissions=["crux_view_all", "crux_executive_dashboard", "crux_forecast_view"]
        )
        
        # Add CRO-specific primitives
        cro_primitives = [
            "revenue_forecast_agent",
            "executive_dashboard_connector",
            "board_reporting_workflow",
            "competitive_analysis_agent",
            "market_expansion_workflow"
        ]
        
        # Create CRO-specific primitives
        self._add_cro_primitives()
        
        cro_library.primitives = cro_primitives
        cro_library.workflows = [
            "executive_revenue_dashboard",
            "quarterly_board_report",
            "market_opportunity_analysis",
            "competitive_intelligence_workflow"
        ]
        
        self.persona_libraries[PersonaType.CRO] = cro_library
    
    def _add_cro_primitives(self):
        """Add CRO-specific primitives to base library"""
        
        # Executive Dashboard Connector
        exec_dashboard = DSLPrimitive(
            primitive_id="executive_dashboard_connector",
            name="Executive Dashboard Connector",
            description="High-level revenue metrics for executive reporting",
            category=PrimitiveCategory.CONNECTOR,
            industry_overlay=IndustryOverlay.SAAS,
            template={
                "type": "query",
                "params": {
                    "source": "fabric",
                    "query": """
                    SELECT 
                        DATE_TRUNC('month', close_date) as month,
                        SUM(amount) as revenue,
                        COUNT(*) as deals_closed,
                        AVG(amount) as avg_deal_size,
                        SUM(CASE WHEN stage_name = 'Closed Won' THEN amount ELSE 0 END) as won_revenue
                    FROM opportunities 
                    WHERE tenant_id = {{tenant_id}}
                    AND close_date >= DATEADD(month, -12, GETDATE())
                    GROUP BY DATE_TRUNC('month', close_date)
                    ORDER BY month DESC
                    """,
                    "parameters": {
                        "tenant_id": "{{tenant_id}}"
                    }
                },
                "governance": {
                    "policy_id": "executive_data_policy",
                    "evidence_capture": True,
                    "executive_access": True
                }
            },
            parameters={
                "tenant_id": {"type": "integer", "required": True}
            },
            tags=["executive", "dashboard", "revenue", "cro"]
        )
        self.base_library.add_custom_primitive(exec_dashboard)
        
        # Board Reporting Workflow
        board_report = DSLPrimitive(
            primitive_id="board_reporting_workflow",
            name="Board Reporting Workflow",
            description="Automated quarterly board report generation",
            category=PrimitiveCategory.WORKFLOW,
            industry_overlay=IndustryOverlay.SAAS,
            template={
                "workflow_id": "{{workflow_id}}",
                "name": "Quarterly Board Report",
                "steps": [
                    {
                        "id": "fetch_quarterly_metrics",
                        "type": "query",
                        "params": {
                            "source": "fabric",
                            "query": "EXEC sp_GetQuarterlyMetrics @tenant_id = {{tenant_id}}, @quarter = '{{quarter}}'"
                        }
                    },
                    {
                        "id": "generate_executive_summary",
                        "type": "agent_call",
                        "params": {
                            "agent_id": "executive_reporting_agent",
                            "context": {"metrics": "{{quarterly_metrics}}"},
                            "tools": ["chart_generator", "pdf_generator"]
                        }
                    },
                    {
                        "id": "distribute_report",
                        "type": "notify",
                        "params": {
                            "channel": "email",
                            "to": ["{{board_email_list}}"],
                            "template": "board_report_template",
                            "attachments": ["{{generated_report}}"]
                        }
                    }
                ]
            },
            parameters={
                "workflow_id": {"type": "string", "required": True},
                "tenant_id": {"type": "integer", "required": True},
                "quarter": {"type": "string", "required": True},
                "board_email_list": {"type": "array", "required": True}
            },
            tags=["board", "reporting", "executive", "quarterly"]
        )
        self.base_library.add_custom_primitive(board_report)
    
    def _create_sales_manager_library(self):
        """Create Sales Manager-specific DSL library"""
        sm_library = PersonaLibrary(
            persona=PersonaType.SALES_MANAGER,
            name="Sales Manager Library",
            description="Team management, pipeline oversight, and performance tracking",
            permissions=["crux_view_team", "crux_edit_team", "crux_assign_leads"]
        )
        
        # Add Sales Manager primitives
        self._add_sales_manager_primitives()
        
        sm_library.primitives = [
            "team_performance_connector",
            "pipeline_hygiene_workflow",
            "quota_tracking_agent",
            "sales_coaching_workflow"
        ]
        
        sm_library.workflows = [
            "weekly_team_review",
            "pipeline_forecast_workflow",
            "performance_improvement_plan",
            "territory_optimization"
        ]
        
        self.persona_libraries[PersonaType.SALES_MANAGER] = sm_library
    
    def _add_sales_manager_primitives(self):
        """Add Sales Manager-specific primitives"""
        
        # Team Performance Connector
        team_perf = DSLPrimitive(
            primitive_id="team_performance_connector",
            name="Team Performance Connector",
            description="Track individual and team sales performance metrics",
            category=PrimitiveCategory.CONNECTOR,
            industry_overlay=IndustryOverlay.SAAS,
            template={
                "type": "query",
                "params": {
                    "source": "salesforce",
                    "query": """
                    SELECT 
                        u.Name as rep_name,
                        u.Id as rep_id,
                        COUNT(o.Id) as opportunities_count,
                        SUM(o.Amount) as pipeline_value,
                        AVG(o.Probability) as avg_probability,
                        SUM(CASE WHEN o.StageName = 'Closed Won' THEN o.Amount ELSE 0 END) as closed_won
                    FROM User u
                    LEFT JOIN Opportunity o ON o.OwnerId = u.Id
                    WHERE u.ManagerId = '{{manager_id}}'
                    AND o.CreatedDate >= {{start_date}}
                    GROUP BY u.Name, u.Id
                    """,
                    "parameters": {
                        "manager_id": "{{manager_id}}",
                        "start_date": "{{start_date|default('THIS_QUARTER')}}"
                    }
                },
                "governance": {
                    "policy_id": "team_data_policy",
                    "evidence_capture": True,
                    "manager_access": True
                }
            },
            parameters={
                "manager_id": {"type": "string", "required": True},
                "start_date": {"type": "string", "default": "THIS_QUARTER"}
            },
            tags=["team", "performance", "sales_manager", "tracking"]
        )
        self.base_library.add_custom_primitive(team_perf)
        
        # Quota Tracking Agent
        quota_agent = DSLPrimitive(
            primitive_id="quota_tracking_agent",
            name="Quota Tracking Agent",
            description="AI agent for quota tracking and attainment analysis",
            category=PrimitiveCategory.AGENT,
            industry_overlay=IndustryOverlay.SAAS,
            template={
                "agent_id": "quota_tracking_agent",
                "name": "Quota Tracking Agent",
                "type": "agent_call",
                "params": {
                    "agent_id": "quota_tracking_agent",
                    "context": {
                        "team_performance": "{{team_performance}}",
                        "quota_targets": "{{quota_targets}}",
                        "time_period": "{{time_period|default('current_quarter')}}"
                    },
                    "tools": ["analytics_tool", "notification_tool", "chart_generator"],
                    "max_depth": 3,
                    "timeout_sec": 120
                },
                "governance": {
                    "policy_id": "quota_policy",
                    "evidence_capture": True,
                    "manager_approval": False
                }
            },
            parameters={
                "team_performance": {"type": "object", "required": True},
                "quota_targets": {"type": "object", "required": True},
                "time_period": {"type": "string", "default": "current_quarter"}
            },
            tags=["quota", "tracking", "ai_agent", "sales_manager"]
        )
        self.base_library.add_custom_primitive(quota_agent)
    
    def _create_revenue_ops_library(self):
        """Create Revenue Operations-specific DSL library"""
        revops_library = PersonaLibrary(
            persona=PersonaType.REVENUE_OPS,
            name="Revenue Operations Library",
            description="Process automation, data analysis, and system optimization",
            permissions=["crux_view_all", "crux_edit_all", "crux_admin_config"]
        )
        
        self._add_revenue_ops_primitives()
        
        revops_library.primitives = [
            "data_quality_connector",
            "process_automation_workflow",
            "system_integration_agent",
            "analytics_pipeline_workflow"
        ]
        
        revops_library.workflows = [
            "data_hygiene_automation",
            "lead_routing_optimization",
            "attribution_analysis",
            "system_health_monitoring"
        ]
        
        self.persona_libraries[PersonaType.REVENUE_OPS] = revops_library
    
    def _add_revenue_ops_primitives(self):
        """Add Revenue Operations-specific primitives"""
        
        # Data Quality Connector
        data_quality = DSLPrimitive(
            primitive_id="data_quality_connector",
            name="Data Quality Connector",
            description="Monitor and report on data quality across revenue systems",
            category=PrimitiveCategory.CONNECTOR,
            industry_overlay=IndustryOverlay.SAAS,
            template={
                "type": "query",
                "params": {
                    "source": "fabric",
                    "query": """
                    WITH data_quality_metrics AS (
                        SELECT 
                            'Opportunities' as table_name,
                            COUNT(*) as total_records,
                            COUNT(CASE WHEN amount IS NULL THEN 1 END) as missing_amount,
                            COUNT(CASE WHEN close_date IS NULL THEN 1 END) as missing_close_date,
                            COUNT(CASE WHEN stage_name IS NULL THEN 1 END) as missing_stage
                        FROM opportunities
                        WHERE tenant_id = {{tenant_id}}
                        UNION ALL
                        SELECT 
                            'Accounts' as table_name,
                            COUNT(*) as total_records,
                            COUNT(CASE WHEN name IS NULL THEN 1 END) as missing_name,
                            COUNT(CASE WHEN industry IS NULL THEN 1 END) as missing_industry,
                            COUNT(CASE WHEN annual_revenue IS NULL THEN 1 END) as missing_revenue
                        FROM accounts
                        WHERE tenant_id = {{tenant_id}}
                    )
                    SELECT * FROM data_quality_metrics
                    """,
                    "parameters": {
                        "tenant_id": "{{tenant_id}}"
                    }
                },
                "governance": {
                    "policy_id": "data_ops_policy",
                    "evidence_capture": True,
                    "ops_access": True
                }
            },
            parameters={
                "tenant_id": {"type": "integer", "required": True}
            },
            tags=["data_quality", "monitoring", "revenue_ops", "automation"]
        )
        self.base_library.add_custom_primitive(data_quality)
    
    def _create_customer_success_library(self):
        """Create Customer Success-specific DSL library"""
        cs_library = PersonaLibrary(
            persona=PersonaType.CUSTOMER_SUCCESS,
            name="Customer Success Library",
            description="Customer health monitoring, retention, and expansion",
            permissions=["crux_view_customers", "crux_edit_customers", "crux_create_tasks"]
        )
        
        self._add_customer_success_primitives()
        
        cs_library.primitives = [
            "customer_health_connector",
            "churn_prevention_agent",
            "expansion_opportunity_workflow",
            "customer_satisfaction_tracker"
        ]
        
        cs_library.workflows = [
            "health_score_monitoring",
            "renewal_risk_assessment",
            "upsell_identification",
            "customer_onboarding_automation"
        ]
        
        self.persona_libraries[PersonaType.CUSTOMER_SUCCESS] = cs_library
    
    def _add_customer_success_primitives(self):
        """Add Customer Success-specific primitives"""
        
        # Customer Health Connector
        health_connector = DSLPrimitive(
            primitive_id="customer_health_connector",
            name="Customer Health Connector",
            description="Comprehensive customer health metrics and scoring",
            category=PrimitiveCategory.CONNECTOR,
            industry_overlay=IndustryOverlay.SAAS,
            template={
                "type": "query",
                "params": {
                    "source": "fabric",
                    "query": """
                    SELECT 
                        a.id as account_id,
                        a.name as account_name,
                        a.annual_revenue,
                        COUNT(DISTINCT c.id) as contact_count,
                        COUNT(DISTINCT o.id) as opportunity_count,
                        AVG(CASE WHEN cs.satisfaction_score IS NOT NULL THEN cs.satisfaction_score ELSE 7 END) as avg_satisfaction,
                        MAX(a.last_activity_date) as last_activity,
                        SUM(CASE WHEN o.stage_name = 'Closed Won' THEN o.amount ELSE 0 END) as total_revenue
                    FROM accounts a
                    LEFT JOIN contacts c ON c.account_id = a.id
                    LEFT JOIN opportunities o ON o.account_id = a.id
                    LEFT JOIN customer_satisfaction cs ON cs.account_id = a.id
                    WHERE a.tenant_id = {{tenant_id}}
                    AND a.type = 'Customer'
                    GROUP BY a.id, a.name, a.annual_revenue
                    """,
                    "parameters": {
                        "tenant_id": "{{tenant_id}}"
                    }
                },
                "governance": {
                    "policy_id": "customer_data_policy",
                    "evidence_capture": True,
                    "customer_access": True
                }
            },
            parameters={
                "tenant_id": {"type": "integer", "required": True}
            },
            tags=["customer_health", "monitoring", "customer_success", "retention"]
        )
        self.base_library.add_custom_primitive(health_connector)
    
    def _create_finance_library(self):
        """Create Finance-specific DSL library"""
        finance_library = PersonaLibrary(
            persona=PersonaType.FINANCE,
            name="Finance Library",
            description="Revenue recognition, compliance, and financial reporting",
            permissions=["crux_view_financial", "crux_edit_financial", "crux_compliance_access"]
        )
        
        self._add_finance_primitives()
        
        finance_library.primitives = [
            "revenue_recognition_connector",
            "compliance_monitoring_agent",
            "financial_reporting_workflow",
            "sox_compliance_checker"
        ]
        
        finance_library.workflows = [
            "monthly_revenue_close",
            "sox_compliance_validation",
            "financial_audit_preparation",
            "revenue_variance_analysis"
        ]
        
        self.persona_libraries[PersonaType.FINANCE] = finance_library
    
    def _add_finance_primitives(self):
        """Add Finance-specific primitives"""
        
        # Revenue Recognition Connector
        rev_rec = DSLPrimitive(
            primitive_id="revenue_recognition_connector",
            name="Revenue Recognition Connector",
            description="ASC 606 compliant revenue recognition calculations",
            category=PrimitiveCategory.CONNECTOR,
            industry_overlay=IndustryOverlay.SAAS,
            template={
                "type": "query",
                "params": {
                    "source": "fabric",
                    "query": """
                    WITH revenue_schedule AS (
                        SELECT 
                            o.id as opportunity_id,
                            o.name as opportunity_name,
                            o.amount as contract_value,
                            o.close_date as contract_start,
                            DATEADD(month, COALESCE(o.contract_term, 12), o.close_date) as contract_end,
                            o.amount / COALESCE(o.contract_term, 12) as monthly_revenue,
                            CASE 
                                WHEN o.revenue_type = 'Subscription' THEN 'Recurring'
                                WHEN o.revenue_type = 'Professional Services' THEN 'Services'
                                ELSE 'One-time'
                            END as revenue_category
                        FROM opportunities o
                        WHERE o.tenant_id = {{tenant_id}}
                        AND o.stage_name = 'Closed Won'
                        AND o.close_date >= '{{start_date}}'
                        AND o.close_date <= '{{end_date}}'
                    )
                    SELECT * FROM revenue_schedule
                    ORDER BY contract_start DESC
                    """,
                    "parameters": {
                        "tenant_id": "{{tenant_id}}",
                        "start_date": "{{start_date}}",
                        "end_date": "{{end_date}}"
                    }
                },
                "governance": {
                    "policy_id": "financial_data_policy",
                    "evidence_capture": True,
                    "sox_compliant": True,
                    "immutable_audit": True
                }
            },
            parameters={
                "tenant_id": {"type": "integer", "required": True},
                "start_date": {"type": "string", "required": True},
                "end_date": {"type": "string", "required": True}
            },
            tags=["revenue_recognition", "asc606", "finance", "compliance"]
        )
        self.base_library.add_custom_primitive(rev_rec)
    
    def get_persona_library(self, persona: PersonaType) -> Optional[PersonaLibrary]:
        """Get library for specific persona"""
        return self.persona_libraries.get(persona)
    
    def get_primitives_for_persona(self, persona: PersonaType) -> List[DSLPrimitive]:
        """Get all primitives available to a persona"""
        persona_lib = self.get_persona_library(persona)
        if not persona_lib:
            return []
        
        primitives = []
        for primitive_id in persona_lib.primitives:
            primitive = self.base_library.get_primitive(primitive_id)
            if primitive:
                primitives.append(primitive)
        
        return primitives
    
    def check_persona_access(self, persona: PersonaType, primitive_id: str, user_permissions: List[str]) -> bool:
        """Check if persona has access to a specific primitive"""
        persona_lib = self.get_persona_library(persona)
        if not persona_lib:
            return False
        
        # Check if primitive is in persona library
        if primitive_id not in persona_lib.primitives:
            return False
        
        # Check permissions
        required_permissions = persona_lib.permissions
        return any(perm in user_permissions for perm in required_permissions)
    
    def export_persona_library(self, persona: PersonaType, format: str = "yaml") -> str:
        """Export a specific persona library"""
        persona_lib = self.get_persona_library(persona)
        if not persona_lib:
            raise ValueError(f"Persona library not found: {persona}")
        
        # Get full primitive definitions
        primitives_data = {}
        for primitive_id in persona_lib.primitives:
            primitive = self.base_library.get_primitive(primitive_id)
            if primitive:
                primitives_data[primitive_id] = {
                    "name": primitive.name,
                    "description": primitive.description,
                    "category": primitive.category.value,
                    "template": primitive.template,
                    "parameters": primitive.parameters,
                    "tags": primitive.tags
                }
        
        export_data = {
            "persona_library": {
                "persona": persona_lib.persona.value,
                "name": persona_lib.name,
                "description": persona_lib.description,
                "permissions": persona_lib.permissions,
                "primitives": primitives_data,
                "workflows": persona_lib.workflows
            }
        }
        
        if format.lower() == "yaml":
            return yaml.dump(export_data, default_flow_style=False)
        elif format.lower() == "json":
            return json.dumps(export_data, indent=2)
        else:
            raise ValueError(f"Unsupported format: {format}")


# Global instance
def create_persona_libraries(base_library: DSLPrimitiveLibrary) -> PersonaSpecificLibraries:
    """Create persona-specific libraries"""
    return PersonaSpecificLibraries(base_library)
