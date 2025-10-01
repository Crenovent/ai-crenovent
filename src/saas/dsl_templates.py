"""
SaaS Industry DSL Templates for Pipeline Hygiene Workflows
Implements Chapter 9.4 requirements for SaaS-specific workflow mappings
"""

from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
from enum import Enum
import yaml
import json

class SaaSWorkflowType(Enum):
    PIPELINE_HYGIENE = "pipeline_hygiene"
    FORECAST_ACCURACY = "forecast_accuracy"
    LEAD_SCORING = "lead_scoring"
    CHURN_PREVENTION = "churn_prevention"
    REVENUE_RECOGNITION = "revenue_recognition"

class SaaSComplianceType(Enum):
    SOX = "sox"
    GDPR = "gdpr"
    DSAR = "dsar"
    CCPA = "ccpa"

class SaaSDSLTemplates:
    """Core DSL templates for SaaS industry workflows"""
    
    @staticmethod
    def pipeline_hygiene_template() -> Dict[str, Any]:
        """
        Core SaaS Pipeline Hygiene workflow template
        Monitors pipeline health, stalled opportunities, and coverage ratios
        """
        return {
            "workflow_id": "saas_pipeline_hygiene_v1.0",
            "name": "SaaS Pipeline Hygiene Monitor",
            "description": "Automated monitoring of sales pipeline health and hygiene metrics",
            "industry": "SaaS",
            "version": "1.0.0",
            "compliance_frameworks": ["SOX", "DSAR"],
            "governance": {
                "sox_compliant": True,
                "audit_enabled": True,
                "override_tracking": True,
                "evidence_collection": True
            },
            "metadata": {
                "created_by": "system",
                "created_at": datetime.utcnow().isoformat(),
                "tags": ["pipeline", "hygiene", "sales", "automation"],
                "priority": "high",
                "execution_frequency": "daily"
            },
            "triggers": [
                {
                    "type": "schedule",
                    "schedule": "0 9 * * MON-FRI",  # Daily at 9 AM, weekdays only
                    "timezone": "UTC"
                },
                {
                    "type": "event",
                    "event_name": "opportunity_stage_change",
                    "conditions": ["stage IN ('negotiation', 'proposal', 'closed_lost')"]
                },
                {
                    "type": "threshold",
                    "metric": "pipeline_coverage_ratio",
                    "operator": "lt",
                    "value": 3.0
                }
            ],
            "parameters": {
                "stalled_opportunity_threshold_days": 15,
                "pipeline_coverage_minimum": 3.0,
                "hygiene_score_threshold": 0.8,
                "notification_cooldown_hours": 24,
                "escalation_threshold_days": 30
            },
            "steps": [
                {
                    "id": "initialize_hygiene_check",
                    "type": "initialization",
                    "name": "Initialize Pipeline Hygiene Check",
                    "description": "Set up context and validate prerequisites",
                    "config": {
                        "validate_data_sources": True,
                        "check_permissions": True,
                        "log_execution_start": True
                    }
                },
                {
                    "id": "query_stalled_opportunities",
                    "type": "data_query",
                    "name": "Identify Stalled Opportunities",
                    "description": "Query for opportunities that haven't progressed in specified timeframe",
                    "depends_on": ["initialize_hygiene_check"],
                    "config": {
                        "query": """
                        SELECT 
                            o.opportunity_id,
                            o.opportunity_name,
                            o.current_stage,
                            o.stage_entry_date,
                            o.amount,
                            o.close_date,
                            o.owner_id,
                            DATEDIFF(CURRENT_DATE, o.stage_entry_date) as days_in_stage,
                            a.name as account_name,
                            u.name as owner_name,
                            u.email as owner_email
                        FROM opportunities o
                        JOIN accounts a ON o.account_id = a.account_id
                        JOIN users u ON o.owner_id = u.user_id
                        WHERE o.stage IN ('qualification', 'needs_analysis', 'proposal', 'negotiation')
                        AND DATEDIFF(CURRENT_DATE, o.stage_entry_date) > {{stalled_opportunity_threshold_days}}
                        AND o.is_closed = FALSE
                        ORDER BY days_in_stage DESC
                        """,
                        "parameters": {
                            "stalled_opportunity_threshold_days": "{{parameters.stalled_opportunity_threshold_days}}"
                        },
                        "max_results": 1000
                    },
                    "output": {
                        "variable": "stalled_opportunities",
                        "schema": "stalled_opportunities_schema"
                    }
                },
                {
                    "id": "calculate_pipeline_metrics",
                    "type": "calculation",
                    "name": "Calculate Pipeline Coverage Metrics",
                    "description": "Calculate pipeline coverage ratio and other key metrics",
                    "depends_on": ["query_stalled_opportunities"],
                    "config": {
                        "calculations": [
                            {
                                "name": "total_pipeline_value",
                                "formula": "SUM(opportunities.amount WHERE stage != 'closed_lost')"
                            },
                            {
                                "name": "quarterly_quota",
                                "formula": "SUM(quotas.amount WHERE quarter = CURRENT_QUARTER)"
                            },
                            {
                                "name": "pipeline_coverage_ratio",
                                "formula": "total_pipeline_value / quarterly_quota"
                            },
                            {
                                "name": "stalled_count",
                                "formula": "COUNT(stalled_opportunities)"
                            },
                            {
                                "name": "stalled_value",
                                "formula": "SUM(stalled_opportunities.amount)"
                            },
                            {
                                "name": "hygiene_score",
                                "formula": "MAX(0, MIN(1, (pipeline_coverage_ratio - stalled_count/100)))"
                            }
                        ]
                    },
                    "output": {
                        "variable": "pipeline_metrics",
                        "schema": "pipeline_metrics_schema"
                    }
                },
                {
                    "id": "evaluate_hygiene_rules",
                    "type": "rule_evaluation",
                    "name": "Evaluate Pipeline Hygiene Rules",
                    "description": "Apply business rules to determine if action is needed",
                    "depends_on": ["calculate_pipeline_metrics"],
                    "config": {
                        "rules": [
                            {
                                "id": "coverage_ratio_check",
                                "name": "Pipeline Coverage Ratio Check",
                                "condition": "pipeline_metrics.pipeline_coverage_ratio >= {{parameters.pipeline_coverage_minimum}}",
                                "severity": "high",
                                "action_required": True
                            },
                            {
                                "id": "stalled_opportunities_check",
                                "name": "Stalled Opportunities Check",
                                "condition": "pipeline_metrics.stalled_count <= 10",
                                "severity": "medium",
                                "action_required": True
                            },
                            {
                                "id": "hygiene_score_check",
                                "name": "Overall Hygiene Score Check",
                                "condition": "pipeline_metrics.hygiene_score >= {{parameters.hygiene_score_threshold}}",
                                "severity": "high",
                                "action_required": True
                            }
                        ]
                    },
                    "output": {
                        "variable": "rule_evaluation_results",
                        "schema": "rule_evaluation_schema"
                    }
                },
                {
                    "id": "generate_hygiene_report",
                    "type": "report_generation",
                    "name": "Generate Pipeline Hygiene Report",
                    "description": "Create detailed report of pipeline hygiene status",
                    "depends_on": ["evaluate_hygiene_rules"],
                    "config": {
                        "report_template": "saas_pipeline_hygiene_report",
                        "include_sections": [
                            "executive_summary",
                            "key_metrics",
                            "stalled_opportunities",
                            "recommendations",
                            "action_items"
                        ],
                        "format": "html",
                        "generate_pdf": True
                    },
                    "output": {
                        "variable": "hygiene_report",
                        "schema": "report_schema"
                    }
                },
                {
                    "id": "send_notifications",
                    "type": "notification",
                    "name": "Send Hygiene Alerts",
                    "description": "Send notifications based on rule evaluation results",
                    "depends_on": ["generate_hygiene_report"],
                    "config": {
                        "notification_rules": [
                            {
                                "condition": "rule_evaluation_results.coverage_ratio_check.failed == true",
                                "recipients": ["sales_ops@company.com", "cro@company.com"],
                                "template": "low_pipeline_coverage_alert",
                                "priority": "high",
                                "channels": ["email", "slack"]
                            },
                            {
                                "condition": "rule_evaluation_results.stalled_opportunities_check.failed == true",
                                "recipients": ["{{stalled_opportunities.owner_email}}"],
                                "template": "stalled_opportunity_reminder",
                                "priority": "medium",
                                "channels": ["email"]
                            },
                            {
                                "condition": "rule_evaluation_results.hygiene_score_check.failed == true",
                                "recipients": ["sales_ops@company.com"],
                                "template": "hygiene_score_alert",
                                "priority": "high",
                                "channels": ["email", "slack", "teams"]
                            }
                        ],
                        "cooldown_hours": "{{parameters.notification_cooldown_hours}}"
                    }
                },
                {
                    "id": "create_follow_up_tasks",
                    "type": "task_creation",
                    "name": "Create Follow-up Tasks",
                    "description": "Automatically create tasks for addressing hygiene issues",
                    "depends_on": ["send_notifications"],
                    "config": {
                        "task_rules": [
                            {
                                "condition": "stalled_opportunities.count > 0",
                                "task_template": {
                                    "title": "Follow up on stalled opportunity: {{opportunity_name}}",
                                    "description": "Opportunity has been in {{current_stage}} for {{days_in_stage}} days",
                                    "assignee": "{{owner_email}}",
                                    "due_date": "+3 days",
                                    "priority": "high"
                                }
                            }
                        ]
                    }
                },
                {
                    "id": "update_audit_trail",
                    "type": "audit_logging",
                    "name": "Update Audit Trail",
                    "description": "Log execution results for compliance and governance",
                    "depends_on": ["create_follow_up_tasks"],
                    "config": {
                        "audit_events": [
                            {
                                "event_type": "workflow_execution",
                                "event_data": {
                                    "workflow_id": "{{workflow_id}}",
                                    "execution_id": "{{execution_id}}",
                                    "metrics": "{{pipeline_metrics}}",
                                    "rules_evaluated": "{{rule_evaluation_results}}",
                                    "notifications_sent": "{{notifications_sent}}",
                                    "tasks_created": "{{tasks_created}}"
                                }
                            }
                        ],
                        "retention_days": 2555,  # 7 years for SOX compliance
                        "encryption": True,
                        "immutable": True
                    }
                }
            ],
            "error_handling": {
                "retry_policy": {
                    "max_retries": 3,
                    "backoff_strategy": "exponential",
                    "retry_on": ["database_connection_error", "api_timeout"]
                },
                "fallback_actions": [
                    {
                        "condition": "data_query_failed",
                        "action": "send_error_notification",
                        "recipients": ["devops@company.com"]
                    }
                ]
            },
            "success_criteria": {
                "execution_time_sla": "300s",  # 5 minutes
                "data_freshness_requirement": "1h",
                "notification_delivery_sla": "60s"
            }
        }
    
    @staticmethod
    def forecast_accuracy_template() -> Dict[str, Any]:
        """SaaS Forecast Accuracy monitoring template"""
        return {
            "workflow_id": "saas_forecast_accuracy_v1.0",
            "name": "SaaS Forecast Accuracy Monitor",
            "description": "Monitor and improve forecast accuracy across sales teams",
            "industry": "SaaS",
            "version": "1.0.0",
            "compliance_frameworks": ["SOX"],
            "triggers": [
                {
                    "type": "schedule",
                    "schedule": "0 0 25 * *",  # 25th of every month
                    "timezone": "UTC"
                }
            ],
            "parameters": {
                "forecast_variance_threshold": 0.15,  # 15% variance
                "accuracy_target": 0.90,  # 90% accuracy
                "historical_periods": 6
            },
            "steps": [
                {
                    "id": "calculate_forecast_accuracy",
                    "type": "calculation",
                    "name": "Calculate Forecast Accuracy",
                    "config": {
                        "calculations": [
                            {
                                "name": "forecast_variance",
                                "formula": "ABS(actual_revenue - forecasted_revenue) / forecasted_revenue"
                            },
                            {
                                "name": "accuracy_score",
                                "formula": "1 - forecast_variance"
                            }
                        ]
                    }
                },
                {
                    "id": "identify_accuracy_issues",
                    "type": "rule_evaluation",
                    "name": "Identify Accuracy Issues",
                    "config": {
                        "rules": [
                            {
                                "condition": "accuracy_score < {{parameters.accuracy_target}}",
                                "action": "flag_for_review"
                            }
                        ]
                    }
                }
            ]
        }
    
    @staticmethod
    def lead_scoring_template() -> Dict[str, Any]:
        """SaaS Lead Scoring automation template"""
        return {
            "workflow_id": "saas_lead_scoring_v1.0",
            "name": "SaaS Lead Scoring Automation",
            "description": "Automated lead scoring based on behavioral and demographic data",
            "industry": "SaaS",
            "version": "1.0.0",
            "compliance_frameworks": ["GDPR", "CCPA"],
            "triggers": [
                {
                    "type": "event",
                    "event_name": "new_lead_created"
                },
                {
                    "type": "event", 
                    "event_name": "lead_activity_updated"
                }
            ],
            "parameters": {
                "demographic_weight": 0.4,
                "behavioral_weight": 0.6,
                "hot_lead_threshold": 80,
                "cold_lead_threshold": 30
            },
            "steps": [
                {
                    "id": "calculate_lead_score",
                    "type": "ml_scoring",
                    "name": "Calculate Lead Score",
                    "config": {
                        "model": "lead_scoring_model_v2",
                        "features": [
                            "company_size",
                            "industry_match",
                            "website_visits",
                            "email_engagement",
                            "content_downloads"
                        ]
                    }
                },
                {
                    "id": "assign_lead_grade",
                    "type": "rule_evaluation",
                    "name": "Assign Lead Grade",
                    "config": {
                        "rules": [
                            {
                                "condition": "lead_score >= {{parameters.hot_lead_threshold}}",
                                "action": "set_grade",
                                "value": "A"
                            },
                            {
                                "condition": "lead_score >= 60 AND lead_score < {{parameters.hot_lead_threshold}}",
                                "action": "set_grade",
                                "value": "B"
                            },
                            {
                                "condition": "lead_score >= {{parameters.cold_lead_threshold}} AND lead_score < 60",
                                "action": "set_grade",
                                "value": "C"
                            },
                            {
                                "condition": "lead_score < {{parameters.cold_lead_threshold}}",
                                "action": "set_grade",
                                "value": "D"
                            }
                        ]
                    }
                }
            ]
        }

class SaaSDSLTemplateManager:
    """Manager for SaaS DSL templates with validation and compilation"""
    
    def __init__(self):
        self.templates = {
            SaaSWorkflowType.PIPELINE_HYGIENE: SaaSDSLTemplates.pipeline_hygiene_template,
            SaaSWorkflowType.FORECAST_ACCURACY: SaaSDSLTemplates.forecast_accuracy_template,
            SaaSWorkflowType.LEAD_SCORING: SaaSDSLTemplates.lead_scoring_template
        }
    
    def get_template(self, workflow_type: SaaSWorkflowType) -> Dict[str, Any]:
        """Get a specific template by type"""
        if workflow_type not in self.templates:
            raise ValueError(f"Template not found for workflow type: {workflow_type}")
        
        return self.templates[workflow_type]()
    
    def list_available_templates(self) -> List[str]:
        """List all available template types"""
        return [wf_type.value for wf_type in self.templates.keys()]
    
    def validate_template(self, template: Dict[str, Any]) -> Dict[str, Any]:
        """Validate template structure and required fields"""
        required_fields = [
            "workflow_id", "name", "industry", "version", 
            "triggers", "parameters", "steps"
        ]
        
        validation_result = {
            "is_valid": True,
            "errors": [],
            "warnings": []
        }
        
        # Check required fields
        for field in required_fields:
            if field not in template:
                validation_result["is_valid"] = False
                validation_result["errors"].append(f"Missing required field: {field}")
        
        # Validate industry is SaaS
        if template.get("industry") != "SaaS":
            validation_result["warnings"].append("Template industry should be 'SaaS'")
        
        # Validate steps structure
        if "steps" in template:
            for i, step in enumerate(template["steps"]):
                if "id" not in step:
                    validation_result["errors"].append(f"Step {i} missing required 'id' field")
                if "type" not in step:
                    validation_result["errors"].append(f"Step {i} missing required 'type' field")
        
        return validation_result
    
    def export_template_as_yaml(self, workflow_type: SaaSWorkflowType, file_path: str):
        """Export template as YAML file"""
        template = self.get_template(workflow_type)
        
        with open(file_path, 'w') as f:
            yaml.dump(template, f, default_flow_style=False, sort_keys=False)
    
    def export_template_as_json(self, workflow_type: SaaSWorkflowType, file_path: str):
        """Export template as JSON file"""
        template = self.get_template(workflow_type)
        
        with open(file_path, 'w') as f:
            json.dump(template, f, indent=2, default=str)

# Example usage and testing
if __name__ == "__main__":
    manager = SaaSDSLTemplateManager()
    
    # Get pipeline hygiene template
    pipeline_template = manager.get_template(SaaSWorkflowType.PIPELINE_HYGIENE)
    print(f"Pipeline Hygiene Template: {pipeline_template['name']}")
    
    # Validate template
    validation_result = manager.validate_template(pipeline_template)
    print(f"Validation Result: {validation_result}")
    
    # List all templates
    available_templates = manager.list_available_templates()
    print(f"Available Templates: {available_templates}")
