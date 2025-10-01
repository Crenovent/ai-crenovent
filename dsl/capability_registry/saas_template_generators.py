#!/usr/bin/env python3
"""
SaaS Template Generators
========================

Dynamic template generators for SaaS automation capabilities.
These generators create adaptive, context-aware templates based on discovered business patterns.

Template Types:
- RBA: Deterministic workflows for standard SaaS processes
- RBIA: ML-augmented workflows for predictive analytics  
- AALA: Agent-led workflows for complex decision making

Each generator creates templates that adapt to:
- Business model (subscription, usage-based, freemium, enterprise)
- Revenue patterns (ARR, MRR, growth trajectories)
- Customer lifecycle (acquisition, retention, expansion, churn)
- Sales processes (pipeline stages, velocity, conversion rates)
"""

import json
import uuid
from datetime import datetime
from typing import Dict, List, Any, Optional
from .dynamic_saas_engine import (
    SaaSDataPattern, DynamicCapabilityTemplate, 
    SaaSBusinessModel, SaaSMetricCategory
)

class SaaSRBATemplateGenerator:
    """Generates Rule-Based Automation templates for SaaS workflows"""
    
    def __init__(self):
        self.template_library = {
            SaaSMetricCategory.REVENUE_METRICS: self._generate_revenue_rba_templates,
            SaaSMetricCategory.GROWTH_METRICS: self._generate_growth_rba_templates,
            SaaSMetricCategory.RETENTION_METRICS: self._generate_retention_rba_templates,
            SaaSMetricCategory.SALES_METRICS: self._generate_sales_rba_templates,
            SaaSMetricCategory.PRODUCT_METRICS: self._generate_product_rba_templates,
            SaaSMetricCategory.CUSTOMER_SUCCESS: self._generate_cs_rba_templates
        }
    
    async def generate_templates(self, tenant_id: int, pattern: SaaSDataPattern) -> List[DynamicCapabilityTemplate]:
        """Generate RBA templates based on discovered pattern"""
        generator_func = self.template_library.get(pattern.pattern_type)
        if generator_func:
            return await generator_func(tenant_id, pattern)
        return []
    
    async def _generate_revenue_rba_templates(self, tenant_id: int, pattern: SaaSDataPattern) -> List[DynamicCapabilityTemplate]:
        """Generate revenue-focused RBA templates"""
        templates = []
        
        # 1. ARR Tracking & Reporting Template
        arr_template = DynamicCapabilityTemplate(
            template_id=str(uuid.uuid4()),
            name=f"Dynamic ARR Tracker - {pattern.business_model.value}",
            description="Automatically tracks Annual Recurring Revenue with dynamic business model adaptation",
            capability_type="RBA_TEMPLATE",
            category="revenue_tracking",
            business_model=pattern.business_model,
            template_definition={
                "workflow_type": "scheduled_reporting",
                "trigger": {
                    "type": "cron",
                    "schedule": "0 9 1 * *",  # First day of each month at 9 AM
                    "timezone": "UTC"
                },
                "steps": [
                    {
                        "id": "fetch_revenue_data",
                        "type": "query",
                        "params": {
                            "data_source": "opportunities",
                            "query": self._generate_arr_query(pattern),
                            "tenant_scoped": True
                        }
                    },
                    {
                        "id": "calculate_arr_metrics",
                        "type": "decision",
                        "params": {
                            "rules": self._generate_arr_calculation_rules(pattern),
                            "business_model": pattern.business_model.value
                        }
                    },
                    {
                        "id": "generate_arr_report",
                        "type": "notify",
                        "params": {
                            "channels": ["slack", "email"],
                            "template": "arr_monthly_report",
                            "recipients": ["cro@company.com", "cfo@company.com"],
                            "format": "executive_dashboard"
                        }
                    },
                    {
                        "id": "store_evidence",
                        "type": "governance",
                        "params": {
                            "evidence_type": "revenue_calculation",
                            "retention_days": 2555,  # 7 years for SOX compliance
                            "compliance_frameworks": ["SOX"]
                        }
                    }
                ]
            },
            input_schema={
                "type": "object",
                "properties": {
                    "reporting_period": {"type": "string", "enum": ["monthly", "quarterly"]},
                    "include_forecast": {"type": "boolean", "default": True},
                    "currency": {"type": "string", "default": "USD"}
                }
            },
            output_schema={
                "type": "object",
                "properties": {
                    "current_arr": {"type": "number"},
                    "arr_growth_rate": {"type": "number"},
                    "arr_by_segment": {"type": "object"},
                    "forecast_confidence": {"type": "number"},
                    "key_insights": {"type": "array", "items": {"type": "string"}}
                }
            },
            usage_patterns={},
            performance_metrics={},
            adaptation_history=[],
            tenant_id=tenant_id,
            created_from_pattern=pattern.pattern_id,
            confidence_score=pattern.confidence_score,
            created_at=datetime.now(),
            updated_at=datetime.now()
        )
        templates.append(arr_template)
        
        # 2. MRR Growth Monitoring Template
        mrr_template = DynamicCapabilityTemplate(
            template_id=str(uuid.uuid4()),
            name=f"Adaptive MRR Monitor - {pattern.business_model.value}",
            description="Monitors Monthly Recurring Revenue with adaptive thresholds based on business patterns",
            capability_type="RBA_TEMPLATE",
            category="growth_monitoring",
            business_model=pattern.business_model,
            template_definition={
                "workflow_type": "real_time_monitoring",
                "trigger": {
                    "type": "data_change",
                    "source": "opportunities",
                    "condition": "stage_change_to_closed_won"
                },
                "steps": [
                    {
                        "id": "calculate_mrr_impact",
                        "type": "query",
                        "params": {
                            "query": self._generate_mrr_impact_query(pattern),
                            "real_time": True
                        }
                    },
                    {
                        "id": "evaluate_growth_thresholds",
                        "type": "decision",
                        "params": {
                            "rules": self._generate_adaptive_growth_rules(pattern),
                            "dynamic_thresholds": True
                        }
                    },
                    {
                        "id": "trigger_alerts",
                        "type": "notify",
                        "params": {
                            "conditional": True,
                            "conditions": ["significant_growth", "growth_slowdown", "negative_growth"],
                            "escalation_rules": self._generate_escalation_rules(pattern)
                        }
                    }
                ]
            },
            input_schema={
                "type": "object",
                "properties": {
                    "monitoring_sensitivity": {"type": "string", "enum": ["low", "medium", "high"]},
                    "alert_thresholds": {"type": "object"},
                    "business_hours_only": {"type": "boolean", "default": False}
                }
            },
            output_schema={
                "type": "object",
                "properties": {
                    "mrr_change": {"type": "number"},
                    "growth_rate": {"type": "number"},
                    "alert_triggered": {"type": "boolean"},
                    "recommended_actions": {"type": "array"}
                }
            },
            usage_patterns={},
            performance_metrics={},
            adaptation_history=[],
            tenant_id=tenant_id,
            created_from_pattern=pattern.pattern_id,
            confidence_score=pattern.confidence_score * 0.9,
            created_at=datetime.now(),
            updated_at=datetime.now()
        )
        templates.append(mrr_template)
        
        # 3. Revenue Recognition Compliance Template
        revenue_compliance_template = DynamicCapabilityTemplate(
            template_id=str(uuid.uuid4()),
            name="Dynamic Revenue Recognition Compliance",
            description="Ensures revenue recognition compliance with adaptive rules for different SaaS models",
            capability_type="RBA_TEMPLATE",
            category="compliance",
            business_model=pattern.business_model,
            template_definition={
                "workflow_type": "compliance_validation",
                "trigger": {
                    "type": "data_change",
                    "source": "contracts",
                    "condition": "contract_signed_or_modified"
                },
                "steps": [
                    {
                        "id": "extract_contract_terms",
                        "type": "query",
                        "params": {
                            "query": "SELECT contract_type, payment_terms, service_start_date, duration FROM contracts WHERE id = $1",
                            "extract_revenue_schedule": True
                        }
                    },
                    {
                        "id": "apply_revenue_recognition_rules",
                        "type": "decision",
                        "params": {
                            "rules": self._generate_revenue_recognition_rules(pattern),
                            "compliance_frameworks": ["ASC 606", "IFRS 15"],
                            "business_model_specific": True
                        }
                    },
                    {
                        "id": "create_revenue_schedule",
                        "type": "governance",
                        "params": {
                            "evidence_type": "revenue_recognition_schedule",
                            "audit_trail": True,
                            "immutable": True
                        }
                    }
                ]
            },
            input_schema={
                "type": "object",
                "properties": {
                    "contract_id": {"type": "string"},
                    "accounting_standard": {"type": "string", "enum": ["ASC 606", "IFRS 15"]},
                    "override_rules": {"type": "object", "default": {}}
                }
            },
            output_schema={
                "type": "object",
                "properties": {
                    "revenue_schedule": {"type": "array"},
                    "compliance_status": {"type": "string"},
                    "audit_trail_id": {"type": "string"},
                    "exceptions": {"type": "array"}
                }
            },
            usage_patterns={},
            performance_metrics={},
            adaptation_history=[],
            tenant_id=tenant_id,
            created_from_pattern=pattern.pattern_id,
            confidence_score=0.95,  # High confidence for compliance
            created_at=datetime.now(),
            updated_at=datetime.now()
        )
        templates.append(revenue_compliance_template)
        
        return templates
    
    async def _generate_sales_rba_templates(self, tenant_id: int, pattern: SaaSDataPattern) -> List[DynamicCapabilityTemplate]:
        """Generate sales process RBA templates"""
        templates = []
        
        # Pipeline Health Monitor Template
        pipeline_health_template = DynamicCapabilityTemplate(
            template_id=str(uuid.uuid4()),
            name="Adaptive Pipeline Health Monitor",
            description="Monitors sales pipeline health with dynamic thresholds based on historical patterns",
            capability_type="RBA_TEMPLATE",
            category="pipeline_management",
            business_model=pattern.business_model,
            template_definition={
                "workflow_type": "health_monitoring",
                "trigger": {
                    "type": "schedule",
                    "frequency": "daily",
                    "time": "08:00"
                },
                "steps": [
                    {
                        "id": "analyze_pipeline_coverage",
                        "type": "query",
                        "params": {
                            "query": self._generate_pipeline_coverage_query(pattern),
                            "dynamic_thresholds": True
                        }
                    },
                    {
                        "id": "evaluate_pipeline_health",
                        "type": "decision",
                        "params": {
                            "rules": self._generate_pipeline_health_rules(pattern),
                            "adaptive_scoring": True
                        }
                    },
                    {
                        "id": "generate_health_report",
                        "type": "notify",
                        "params": {
                            "template": "pipeline_health_dashboard",
                            "recipients": ["sales_manager", "revops_team"],
                            "include_recommendations": True
                        }
                    }
                ]
            },
            input_schema={
                "type": "object",
                "properties": {
                    "coverage_target": {"type": "number", "default": 3.0},
                    "time_horizon": {"type": "string", "enum": ["30d", "60d", "90d"], "default": "90d"},
                    "include_forecasting": {"type": "boolean", "default": True}
                }
            },
            output_schema={
                "type": "object",
                "properties": {
                    "pipeline_health_score": {"type": "number"},
                    "coverage_ratio": {"type": "number"},
                    "at_risk_deals": {"type": "array"},
                    "recommendations": {"type": "array"}
                }
            },
            usage_patterns={},
            performance_metrics={},
            adaptation_history=[],
            tenant_id=tenant_id,
            created_from_pattern=pattern.pattern_id,
            confidence_score=pattern.confidence_score,
            created_at=datetime.now(),
            updated_at=datetime.now()
        )
        templates.append(pipeline_health_template)
        
        return templates
    
    async def _generate_retention_rba_templates(self, tenant_id: int, pattern: SaaSDataPattern) -> List[DynamicCapabilityTemplate]:
        """Generate customer retention RBA templates"""
        templates = []
        
        # Churn Risk Alert Template
        churn_alert_template = DynamicCapabilityTemplate(
            template_id=str(uuid.uuid4()),
            name="Dynamic Churn Risk Alerting",
            description="Identifies and alerts on churn risk with adaptive scoring based on customer patterns",
            capability_type="RBA_TEMPLATE",
            category="customer_retention",
            business_model=pattern.business_model,
            template_definition={
                "workflow_type": "risk_monitoring",
                "trigger": {
                    "type": "data_change",
                    "source": "customer_interactions",
                    "condition": "engagement_score_change"
                },
                "steps": [
                    {
                        "id": "calculate_churn_indicators",
                        "type": "query",
                        "params": {
                            "query": self._generate_churn_indicators_query(pattern),
                            "include_behavioral_signals": True
                        }
                    },
                    {
                        "id": "evaluate_churn_risk",
                        "type": "decision",
                        "params": {
                            "rules": self._generate_churn_risk_rules(pattern),
                            "risk_scoring": "dynamic"
                        }
                    },
                    {
                        "id": "trigger_retention_actions",
                        "type": "notify",
                        "params": {
                            "conditional": True,
                            "actions": ["alert_csm", "create_retention_task", "escalate_to_management"],
                            "urgency_based_routing": True
                        }
                    }
                ]
            },
            input_schema={
                "type": "object",
                "properties": {
                    "risk_threshold": {"type": "string", "enum": ["low", "medium", "high"]},
                    "monitoring_frequency": {"type": "string", "enum": ["real_time", "daily", "weekly"]},
                    "auto_actions_enabled": {"type": "boolean", "default": True}
                }
            },
            output_schema={
                "type": "object",
                "properties": {
                    "churn_risk_score": {"type": "number"},
                    "risk_factors": {"type": "array"},
                    "recommended_actions": {"type": "array"},
                    "urgency_level": {"type": "string"}
                }
            },
            usage_patterns={},
            performance_metrics={},
            adaptation_history=[],
            tenant_id=tenant_id,
            created_from_pattern=pattern.pattern_id,
            confidence_score=pattern.confidence_score * 0.85,
            created_at=datetime.now(),
            updated_at=datetime.now()
        )
        templates.append(churn_alert_template)
        
        return templates
    
    # Helper methods for generating dynamic queries and rules
    def _generate_arr_query(self, pattern: SaaSDataPattern) -> str:
        """Generate dynamic ARR calculation query based on business model"""
        base_query = """
        SELECT 
            SUM(CASE 
                WHEN contract_type = 'annual' THEN amount
                WHEN contract_type = 'monthly' THEN amount * 12
                WHEN contract_type = 'quarterly' THEN amount * 4
                ELSE amount
            END) as arr,
            COUNT(DISTINCT account_id) as customer_count,
            DATE_TRUNC('month', close_date) as period
        FROM opportunities 
        WHERE tenant_id = $1 
          AND stage_name = 'Closed Won'
          AND close_date >= NOW() - INTERVAL '12 months'
        GROUP BY DATE_TRUNC('month', close_date)
        ORDER BY period
        """
        
        if pattern.business_model == SaaSBusinessModel.USAGE_BASED:
            # Modify query for usage-based model
            base_query = base_query.replace("amount", "COALESCE(usage_amount, amount)")
        elif pattern.business_model == SaaSBusinessModel.FREEMIUM_CONVERSION:
            # Add conversion tracking
            base_query += " AND account_type != 'freemium'"
        
        return base_query
    
    def _generate_arr_calculation_rules(self, pattern: SaaSDataPattern) -> List[Dict[str, Any]]:
        """Generate adaptive ARR calculation rules"""
        rules = [
            {
                "name": "calculate_growth_rate",
                "condition": "current_period_arr > 0 AND previous_period_arr > 0",
                "action": "SET growth_rate = ((current_period_arr - previous_period_arr) / previous_period_arr) * 100"
            },
            {
                "name": "flag_significant_change",
                "condition": f"ABS(growth_rate) > {self._get_growth_threshold(pattern)}",
                "action": "SET requires_attention = true"
            }
        ]
        
        if pattern.business_model == SaaSBusinessModel.ENTERPRISE_CONTRACT:
            rules.append({
                "name": "account_for_large_deals",
                "condition": "individual_deal_size > (average_deal_size * 3)",
                "action": "SET outlier_impact = true"
            })
        
        return rules
    
    def _get_growth_threshold(self, pattern: SaaSDataPattern) -> float:
        """Get adaptive growth threshold based on business maturity"""
        predictability = pattern.key_metrics.get("revenue_predictability", 0.5)
        
        if predictability > 0.8:  # Mature, predictable business
            return 10.0  # 10% change is significant
        elif predictability > 0.5:  # Growing business
            return 20.0  # 20% change is significant
        else:  # Early stage, volatile
            return 50.0  # 50% change is significant
    
    def _generate_mrr_impact_query(self, pattern: SaaSDataPattern) -> str:
        """Generate MRR impact calculation query"""
        return """
        WITH monthly_revenue AS (
            SELECT 
                DATE_TRUNC('month', close_date) as month,
                SUM(CASE 
                    WHEN contract_type = 'monthly' THEN amount
                    WHEN contract_type = 'annual' THEN amount / 12
                    WHEN contract_type = 'quarterly' THEN amount / 3
                    ELSE amount
                END) as mrr
            FROM opportunities 
            WHERE tenant_id = $1 
              AND stage_name = 'Closed Won'
              AND close_date >= NOW() - INTERVAL '6 months'
            GROUP BY DATE_TRUNC('month', close_date)
        )
        SELECT 
            month,
            mrr,
            LAG(mrr) OVER (ORDER BY month) as previous_mrr,
            (mrr - LAG(mrr) OVER (ORDER BY month)) / NULLIF(LAG(mrr) OVER (ORDER BY month), 0) * 100 as growth_rate
        FROM monthly_revenue
        ORDER BY month DESC
        LIMIT 1
        """
    
    def _generate_adaptive_growth_rules(self, pattern: SaaSDataPattern) -> List[Dict[str, Any]]:
        """Generate adaptive growth monitoring rules"""
        base_threshold = self._get_growth_threshold(pattern)
        
        return [
            {
                "name": "significant_growth",
                "condition": f"growth_rate > {base_threshold}",
                "action": "TRIGGER alert WITH priority = 'high' AND message = 'Significant MRR growth detected'"
            },
            {
                "name": "growth_slowdown", 
                "condition": f"growth_rate < {base_threshold * 0.3} AND growth_rate > 0",
                "action": "TRIGGER alert WITH priority = 'medium' AND message = 'MRR growth slowing'"
            },
            {
                "name": "negative_growth",
                "condition": "growth_rate < 0",
                "action": "TRIGGER alert WITH priority = 'critical' AND message = 'MRR decline detected'"
            }
        ]
    
    def _generate_escalation_rules(self, pattern: SaaSDataPattern) -> Dict[str, Any]:
        """Generate escalation rules based on business model"""
        base_rules = {
            "high": ["sales_manager", "revops_lead"],
            "critical": ["sales_manager", "revops_lead", "cro", "cfo"]
        }
        
        if pattern.business_model == SaaSBusinessModel.ENTERPRISE_CONTRACT:
            base_rules["critical"].append("account_executive")
        
        return base_rules
    
    def _generate_revenue_recognition_rules(self, pattern: SaaSDataPattern) -> List[Dict[str, Any]]:
        """Generate revenue recognition rules based on business model"""
        rules = []
        
        if pattern.business_model == SaaSBusinessModel.SUBSCRIPTION_RECURRING:
            rules.extend([
                {
                    "name": "monthly_subscription_recognition",
                    "condition": "contract_type = 'monthly'",
                    "action": "RECOGNIZE revenue_amount OVER contract_duration MONTHLY"
                },
                {
                    "name": "annual_subscription_recognition", 
                    "condition": "contract_type = 'annual'",
                    "action": "RECOGNIZE revenue_amount OVER 12 MONTHLY starting service_start_date"
                }
            ])
        elif pattern.business_model == SaaSBusinessModel.USAGE_BASED:
            rules.append({
                "name": "usage_based_recognition",
                "condition": "contract_type = 'usage'",
                "action": "RECOGNIZE usage_amount WHEN service_delivered"
            })
        
        return rules
    
    def _generate_pipeline_coverage_query(self, pattern: SaaSDataPattern) -> str:
        """Generate pipeline coverage analysis query"""
        return """
        WITH pipeline_data AS (
            SELECT 
                SUM(CASE WHEN stage_name NOT IN ('Closed Won', 'Closed Lost') THEN amount ELSE 0 END) as open_pipeline,
                SUM(CASE WHEN stage_name = 'Closed Won' AND close_date >= DATE_TRUNC('quarter', NOW()) THEN amount ELSE 0 END) as qtd_closed,
                (DATE_TRUNC('quarter', NOW()) + INTERVAL '3 months' - INTERVAL '1 day')::date as quarter_end
            FROM opportunities 
            WHERE tenant_id = $1
        )
        SELECT 
            open_pipeline,
            qtd_closed,
            (SELECT AVG(amount) FROM opportunities WHERE tenant_id = $1 AND stage_name = 'Closed Won' AND close_date >= NOW() - INTERVAL '12 months') as avg_monthly_close,
            open_pipeline / NULLIF(avg_monthly_close * 3, 0) as coverage_ratio
        FROM pipeline_data
        """
    
    def _generate_pipeline_health_rules(self, pattern: SaaSDataPattern) -> List[Dict[str, Any]]:
        """Generate pipeline health evaluation rules"""
        target_coverage = 3.0  # Default 3x coverage
        
        # Adjust based on business model predictability
        predictability = pattern.key_metrics.get("acquisition_predictability", 0.5)
        if predictability > 0.8:
            target_coverage = 2.5  # Lower coverage needed for predictable business
        elif predictability < 0.3:
            target_coverage = 4.0  # Higher coverage needed for unpredictable business
        
        return [
            {
                "name": "healthy_coverage",
                "condition": f"coverage_ratio >= {target_coverage}",
                "action": f"SET health_score = 100 AND status = 'healthy'"
            },
            {
                "name": "warning_coverage",
                "condition": f"coverage_ratio >= {target_coverage * 0.7} AND coverage_ratio < {target_coverage}",
                "action": "SET health_score = 70 AND status = 'warning'"
            },
            {
                "name": "critical_coverage",
                "condition": f"coverage_ratio < {target_coverage * 0.7}",
                "action": "SET health_score = 30 AND status = 'critical'"
            }
        ]
    
    def _generate_churn_indicators_query(self, pattern: SaaSDataPattern) -> str:
        """Generate churn risk indicators query"""
        return """
        SELECT 
            account_id,
            account_name,
            last_login_date,
            support_tickets_30d,
            contract_renewal_date,
            usage_trend_30d,
            payment_issues_count,
            engagement_score,
            CASE 
                WHEN last_login_date < NOW() - INTERVAL '30 days' THEN 1 
                ELSE 0 
            END as login_risk,
            CASE 
                WHEN support_tickets_30d > 5 THEN 1 
                ELSE 0 
            END as support_risk,
            CASE 
                WHEN contract_renewal_date < NOW() + INTERVAL '60 days' THEN 1 
                ELSE 0 
            END as renewal_risk
        FROM accounts 
        WHERE tenant_id = $1 
          AND account_type = 'Customer'
          AND status = 'Active'
        """
    
    def _generate_churn_risk_rules(self, pattern: SaaSDataPattern) -> List[Dict[str, Any]]:
        """Generate churn risk evaluation rules"""
        return [
            {
                "name": "high_churn_risk",
                "condition": "(login_risk + support_risk + renewal_risk) >= 2",
                "action": "SET churn_risk_score = 0.8 AND urgency = 'high'"
            },
            {
                "name": "medium_churn_risk", 
                "condition": "(login_risk + support_risk + renewal_risk) = 1",
                "action": "SET churn_risk_score = 0.5 AND urgency = 'medium'"
            },
            {
                "name": "low_churn_risk",
                "condition": "(login_risk + support_risk + renewal_risk) = 0",
                "action": "SET churn_risk_score = 0.2 AND urgency = 'low'"
            }
        ]
    
    # Additional template generators for other categories...
    async def _generate_growth_rba_templates(self, tenant_id: int, pattern: SaaSDataPattern) -> List[DynamicCapabilityTemplate]:
        """Generate growth-focused RBA templates"""
        # Implementation for growth templates
        return []
    
    async def _generate_product_rba_templates(self, tenant_id: int, pattern: SaaSDataPattern) -> List[DynamicCapabilityTemplate]:
        """Generate product usage RBA templates"""
        # Implementation for product templates
        return []
    
    async def _generate_cs_rba_templates(self, tenant_id: int, pattern: SaaSDataPattern) -> List[DynamicCapabilityTemplate]:
        """Generate customer success RBA templates"""
        # Implementation for customer success templates
        return []


class SaaSRBIATemplateGenerator:
    """Generates Rule-Based Intelligent Automation templates with ML components"""
    
    def __init__(self):
        self.ml_models = {
            "churn_prediction": "customer_churn_classifier",
            "lead_scoring": "lead_conversion_predictor", 
            "revenue_forecasting": "arr_forecast_model",
            "expansion_prediction": "account_expansion_classifier"
        }
    
    async def generate_templates(self, tenant_id: int, pattern: SaaSDataPattern) -> List[DynamicCapabilityTemplate]:
        """Generate RBIA templates with ML augmentation"""
        templates = []
        
        # Intelligent Lead Scoring Template
        lead_scoring_template = DynamicCapabilityTemplate(
            template_id=str(uuid.uuid4()),
            name="AI-Powered Lead Scoring Engine",
            description="Machine learning-augmented lead scoring with adaptive model training",
            capability_type="RBIA_MODEL",
            category="lead_intelligence",
            business_model=pattern.business_model,
            template_definition={
                "workflow_type": "ml_augmented_scoring",
                "trigger": {
                    "type": "data_change",
                    "source": "leads",
                    "condition": "new_lead_created"
                },
                "steps": [
                    {
                        "id": "extract_lead_features",
                        "type": "query",
                        "params": {
                            "query": self._generate_feature_extraction_query(pattern),
                            "feature_engineering": True
                        }
                    },
                    {
                        "id": "ml_lead_scoring",
                        "type": "ml_decision",
                        "params": {
                            "model_name": "lead_conversion_predictor",
                            "confidence_threshold": 0.7,
                            "fallback_to_rules": True,
                            "model_version": "adaptive"
                        }
                    },
                    {
                        "id": "apply_business_rules",
                        "type": "decision",
                        "params": {
                            "rules": self._generate_lead_scoring_rules(pattern),
                            "combine_with_ml": True
                        }
                    },
                    {
                        "id": "route_qualified_leads",
                        "type": "notify",
                        "params": {
                            "conditional_routing": True,
                            "score_thresholds": {"hot": 0.8, "warm": 0.6, "cold": 0.4}
                        }
                    }
                ]
            },
            input_schema={
                "type": "object",
                "properties": {
                    "lead_id": {"type": "string"},
                    "scoring_model": {"type": "string", "default": "adaptive"},
                    "confidence_threshold": {"type": "number", "default": 0.7}
                }
            },
            output_schema={
                "type": "object",
                "properties": {
                    "lead_score": {"type": "number"},
                    "confidence": {"type": "number"},
                    "scoring_factors": {"type": "array"},
                    "recommended_action": {"type": "string"},
                    "model_version": {"type": "string"}
                }
            },
            usage_patterns={},
            performance_metrics={},
            adaptation_history=[],
            tenant_id=tenant_id,
            created_from_pattern=pattern.pattern_id,
            confidence_score=0.85,
            created_at=datetime.now(),
            updated_at=datetime.now()
        )
        templates.append(lead_scoring_template)
        
        return templates
    
    def _generate_feature_extraction_query(self, pattern: SaaSDataPattern) -> str:
        """Generate ML feature extraction query"""
        return """
        SELECT 
            l.id as lead_id,
            l.company_size,
            l.industry,
            l.lead_source,
            l.engagement_score,
            a.annual_revenue,
            a.employee_count,
            COALESCE(similar_customers.conversion_rate, 0) as similar_customer_conversion,
            EXTRACT(DAYS FROM NOW() - l.created_date) as lead_age_days
        FROM leads l
        LEFT JOIN accounts a ON l.company = a.name
        LEFT JOIN (
            SELECT 
                industry,
                AVG(CASE WHEN converted = true THEN 1.0 ELSE 0.0 END) as conversion_rate
            FROM leads 
            WHERE tenant_id = $1 
            GROUP BY industry
        ) similar_customers ON l.industry = similar_customers.industry
        WHERE l.id = $2
        """
    
    def _generate_lead_scoring_rules(self, pattern: SaaSDataPattern) -> List[Dict[str, Any]]:
        """Generate business rules to combine with ML scoring"""
        rules = [
            {
                "name": "enterprise_boost",
                "condition": "company_size = 'Enterprise' AND annual_revenue > 100000000",
                "action": "BOOST score BY 0.2"
            },
            {
                "name": "high_engagement_boost",
                "condition": "engagement_score > 80",
                "action": "BOOST score BY 0.15"
            },
            {
                "name": "referral_boost",
                "condition": "lead_source = 'Referral'",
                "action": "BOOST score BY 0.1"
            }
        ]
        
        # Add business model specific rules
        if pattern.business_model == SaaSBusinessModel.ENTERPRISE_CONTRACT:
            rules.append({
                "name": "enterprise_fit",
                "condition": "employee_count > 1000",
                "action": "BOOST score BY 0.25"
            })
        
        return rules


class SaaSAALATemplateGenerator:
    """Generates AI Agent-Led Automation templates for complex SaaS workflows"""
    
    async def generate_templates(self, tenant_id: int, pattern: SaaSDataPattern) -> List[DynamicCapabilityTemplate]:
        """Generate AALA templates for autonomous SaaS workflows"""
        templates = []
        
        # Conversational Revenue Assistant Template
        revenue_assistant_template = DynamicCapabilityTemplate(
            template_id=str(uuid.uuid4()),
            name="AI Revenue Intelligence Assistant",
            description="Conversational AI agent for revenue analysis, forecasting, and strategic insights",
            capability_type="AALA_AGENT",
            category="revenue_intelligence",
            business_model=pattern.business_model,
            template_definition={
                "workflow_type": "conversational_agent",
                "agent_capabilities": [
                    "revenue_analysis", "forecast_generation", "trend_identification",
                    "scenario_planning", "risk_assessment", "strategic_recommendations"
                ],
                "knowledge_sources": [
                    "opportunities", "accounts", "contracts", "market_data",
                    "historical_performance", "industry_benchmarks"
                ],
                "reasoning_framework": {
                    "type": "multi_step_reasoning",
                    "memory_enabled": True,
                    "context_window": "conversation_session",
                    "tool_access": ["sql_query", "calculation_engine", "visualization_generator"]
                },
                "safety_guardrails": {
                    "financial_impact_threshold": 100000,
                    "require_human_approval": ["forecast_changes", "strategic_recommendations"],
                    "audit_all_decisions": True
                }
            },
            input_schema={
                "type": "object",
                "properties": {
                    "user_query": {"type": "string"},
                    "context": {"type": "object"},
                    "conversation_history": {"type": "array"},
                    "user_role": {"type": "string", "enum": ["CRO", "CFO", "RevOps", "Sales_Manager"]}
                }
            },
            output_schema={
                "type": "object",
                "properties": {
                    "response": {"type": "string"},
                    "insights": {"type": "array"},
                    "visualizations": {"type": "array"},
                    "recommended_actions": {"type": "array"},
                    "confidence_score": {"type": "number"},
                    "requires_approval": {"type": "boolean"}
                }
            },
            usage_patterns={},
            performance_metrics={},
            adaptation_history=[],
            tenant_id=tenant_id,
            created_from_pattern=pattern.pattern_id,
            confidence_score=0.75,
            created_at=datetime.now(),
            updated_at=datetime.now()
        )
        templates.append(revenue_assistant_template)
        
        return templates
