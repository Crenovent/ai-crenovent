"""
Industry-Specific Template System for RBIA
Tasks 4.2.3-4.2.12: Create industry-specific templates with ML models
"""

import json
import uuid
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import logging
import yaml

logger = logging.getLogger(__name__)

@dataclass
class TemplateConfig:
    """Configuration for industry-specific templates"""
    template_id: str
    template_name: str
    industry: str
    template_type: str  # 'churn_risk', 'fraud_detection', 'credit_scoring', etc.
    description: str
    ml_models: List[Dict[str, Any]]
    workflow_steps: List[Dict[str, Any]]
    governance_config: Dict[str, Any]
    explainability_config: Dict[str, Any]
    confidence_thresholds: Dict[str, float]
    sample_data_schema: Dict[str, Any]
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.utcnow)
    version: str = "1.0.0"

class IndustryTemplateRegistry:
    """
    Registry for industry-specific RBIA templates
    
    Provides:
    - Template creation and management
    - Industry-specific ML model configurations
    - Workflow template generation
    - Explainability integration
    - Confidence threshold enforcement
    """
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.templates: Dict[str, TemplateConfig] = {}
        self._initialize_industry_templates()
    
    def _initialize_industry_templates(self):
        """Initialize all industry-specific templates"""
        try:
            # SaaS Templates
            self._create_saas_templates()
            
            # Banking Templates
            self._create_banking_templates()
            
            # Insurance Templates
            self._create_insurance_templates()
            
            # E-commerce Templates
            self._create_ecommerce_templates()
            
            # Financial Services Templates
            self._create_financial_services_templates()
            
            self.logger.info(f"Initialized {len(self.templates)} industry templates")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize industry templates: {e}")
            raise
    
    def _create_saas_templates(self):
        """Create SaaS industry templates"""
        
        # SaaS: Churn Risk Alert Template
        churn_risk_template = TemplateConfig(
            template_id="saas_churn_risk_alert",
            template_name="SaaS Churn Risk Alert",
            industry="SaaS",
            template_type="churn_risk",
            description="Automated churn risk detection and alert system for SaaS customers",
            ml_models=[
                {
                    "model_id": "saas_churn_predictor_v3",
                    "model_name": "SaaS Churn Predictor v3",
                    "model_type": "predict",
                    "input_features": [
                        "customer_tenure_months",
                        "mrr",
                        "usage_frequency",
                        "support_tickets_last_30d",
                        "last_login_days_ago",
                        "feature_adoption_score",
                        "payment_method",
                        "contract_type"
                    ],
                    "output_features": [
                        "churn_probability",
                        "churn_segment",
                        "churn_reason_codes"
                    ],
                    "confidence_threshold": 0.75,
                    "explainability_enabled": True
                },
                {
                    "model_id": "saas_engagement_scorer_v2",
                    "model_name": "SaaS Engagement Scorer v2",
                    "model_type": "score",
                    "input_features": [
                        "login_frequency",
                        "feature_usage_depth",
                        "collaboration_score",
                        "support_interaction_sentiment"
                    ],
                    "output_features": ["engagement_score"],
                    "confidence_threshold": 0.65,
                    "explainability_enabled": False
                }
            ],
            workflow_steps=[
                {
                    "id": "fetch_customer_metrics",
                    "type": "query",
                    "params": {
                        "source": "analytics_db",
                        "query": """
                            SELECT 
                                customer_id,
                                customer_tenure_months,
                                mrr,
                                usage_frequency,
                                support_tickets_last_30d,
                                last_login_days_ago,
                                feature_adoption_score,
                                payment_method,
                                contract_type,
                                login_frequency,
                                feature_usage_depth,
                                collaboration_score,
                                support_interaction_sentiment
                            FROM customer_analytics 
                            WHERE customer_id = {{ customer_id }}
                        """
                    }
                },
                {
                    "id": "predict_churn_risk",
                    "type": "ml_predict",
                    "params": {
                        "model_id": "saas_churn_predictor_v3",
                        "input_data_map": {
                            "customer_tenure_months": "customer_metrics.customer_tenure_months",
                            "mrr": "customer_metrics.mrr",
                            "usage_frequency": "customer_metrics.usage_frequency",
                            "support_tickets_last_30d": "customer_metrics.support_tickets_last_30d",
                            "last_login_days_ago": "customer_metrics.last_login_days_ago",
                            "feature_adoption_score": "customer_metrics.feature_adoption_score",
                            "payment_method": "customer_metrics.payment_method",
                            "contract_type": "customer_metrics.contract_type"
                        },
                        "confidence_threshold": 0.75,
                        "explainability_enabled": True
                    }
                },
                {
                    "id": "score_engagement",
                    "type": "ml_score",
                    "params": {
                        "model_id": "saas_engagement_scorer_v2",
                        "input_data_map": {
                            "login_frequency": "customer_metrics.login_frequency",
                            "feature_usage_depth": "customer_metrics.feature_usage_depth",
                            "collaboration_score": "customer_metrics.collaboration_score",
                            "support_interaction_sentiment": "customer_metrics.support_interaction_sentiment"
                        },
                        "confidence_threshold": 0.65
                    }
                },
                {
                    "id": "evaluate_churn_risk",
                    "type": "decision",
                    "params": {
                        "expression": "churn_probability > 0.7 and engagement_score < 0.4",
                        "on_true": "trigger_high_risk_alert",
                        "on_false": "check_medium_risk"
                    }
                },
                {
                    "id": "check_medium_risk",
                    "type": "decision",
                    "params": {
                        "expression": "churn_probability > 0.5 and engagement_score < 0.6",
                        "on_true": "trigger_medium_risk_alert",
                        "on_false": "log_low_risk"
                    }
                },
                {
                    "id": "trigger_high_risk_alert",
                    "type": "notify",
                    "params": {
                        "channel": "slack",
                        "webhook_url": "{{ slack_webhook_url }}",
                        "message": "🚨 HIGH CHURN RISK: Customer {{ customer_id }} has {{ churn_probability | round(2) }}% churn probability. Engagement: {{ engagement_score | round(2) }}. Reasons: {{ churn_reason_codes | join(', ') }}",
                        "priority": "high"
                    }
                },
                {
                    "id": "trigger_medium_risk_alert",
                    "type": "notify",
                    "params": {
                        "channel": "email",
                        "recipient": "customer-success@company.com",
                        "subject": "Medium Churn Risk Alert - Customer {{ customer_id }}",
                        "body": "Customer {{ customer_id }} shows medium churn risk ({{ churn_probability | round(2) }}%). Consider proactive engagement.",
                        "priority": "medium"
                    }
                },
                {
                    "id": "log_low_risk",
                    "type": "governance",
                    "params": {
                        "action": "log_event",
                        "event_type": "churn_risk_low",
                        "details": {
                            "customer_id": "{{ customer_id }}",
                            "churn_probability": "{{ churn_probability }}",
                            "engagement_score": "{{ engagement_score }}"
                        }
                    }
                }
            ],
            governance_config={
                "policy_pack_id": "saas_churn_governance_v1",
                "trust_score_threshold": 0.75,
                "evidence_pack_required": True,
                "audit_logging": True,
                "bias_monitoring": True,
                "drift_monitoring": True
            },
            explainability_config={
                "enabled": True,
                "methods": ["SHAP", "reason_codes"],
                "feature_importance_tracking": True,
                "explanation_storage": True
            },
            confidence_thresholds={
                "churn_prediction": 0.75,
                "engagement_score": 0.65,
                "overall_confidence": 0.70
            },
            sample_data_schema={
                "customer_id": "string",
                "customer_tenure_months": "integer",
                "mrr": "float",
                "usage_frequency": "float",
                "support_tickets_last_30d": "integer",
                "last_login_days_ago": "integer",
                "feature_adoption_score": "float",
                "payment_method": "string",
                "contract_type": "string",
                "login_frequency": "float",
                "feature_usage_depth": "float",
                "collaboration_score": "float",
                "support_interaction_sentiment": "float"
            }
        )
        
        # SaaS: Forecast Variance Detector Template
        forecast_variance_template = TemplateConfig(
            template_id="saas_forecast_variance_detector",
            template_name="SaaS Forecast Variance Detector",
            industry="SaaS",
            template_type="forecast_variance",
            description="Detects significant variances between forecasted and actual SaaS metrics",
            ml_models=[
                {
                    "model_id": "saas_forecast_anomaly_detector_v1",
                    "model_name": "SaaS Forecast Anomaly Detector v1",
                    "model_type": "classify",
                    "input_features": [
                        "forecasted_mrr",
                        "actual_mrr",
                        "forecasted_new_customers",
                        "actual_new_customers",
                        "forecasted_churn_rate",
                        "actual_churn_rate",
                        "seasonal_factor",
                        "market_conditions"
                    ],
                    "output_features": [
                        "variance_category",
                        "variance_severity",
                        "variance_reasons"
                    ],
                    "confidence_threshold": 0.70,
                    "explainability_enabled": True
                }
            ],
            workflow_steps=[
                {
                    "id": "fetch_forecast_data",
                    "type": "query",
                    "params": {
                        "source": "forecast_db",
                        "query": """
                            SELECT 
                                period_id,
                                forecasted_mrr,
                                actual_mrr,
                                forecasted_new_customers,
                                actual_new_customers,
                                forecasted_churn_rate,
                                actual_churn_rate,
                                seasonal_factor,
                                market_conditions
                            FROM forecast_actuals 
                            WHERE period_id = {{ period_id }}
                        """
                    }
                },
                {
                    "id": "detect_variance_anomaly",
                    "type": "ml_classify",
                    "params": {
                        "model_id": "saas_forecast_anomaly_detector_v1",
                        "input_data_map": {
                            "forecasted_mrr": "forecast_data.forecasted_mrr",
                            "actual_mrr": "forecast_data.actual_mrr",
                            "forecasted_new_customers": "forecast_data.forecasted_new_customers",
                            "actual_new_customers": "forecast_data.actual_new_customers",
                            "forecasted_churn_rate": "forecast_data.forecasted_churn_rate",
                            "actual_churn_rate": "forecast_data.actual_churn_rate",
                            "seasonal_factor": "forecast_data.seasonal_factor",
                            "market_conditions": "forecast_data.market_conditions"
                        },
                        "classes": ["normal_variance", "significant_variance", "critical_variance"],
                        "confidence_threshold": 0.70,
                        "explainability_enabled": True
                    }
                },
                {
                    "id": "evaluate_variance_severity",
                    "type": "decision",
                    "params": {
                        "expression": "variance_category == 'critical_variance'",
                        "on_true": "trigger_critical_alert",
                        "on_false": "check_significant_variance"
                    }
                },
                {
                    "id": "check_significant_variance",
                    "type": "decision",
                    "params": {
                        "expression": "variance_category == 'significant_variance'",
                        "on_true": "trigger_significant_alert",
                        "on_false": "log_normal_variance"
                    }
                },
                {
                    "id": "trigger_critical_alert",
                    "type": "notify",
                    "params": {
                        "channel": "pagerduty",
                        "integration_key": "{{ pagerduty_key }}",
                        "message": "🚨 CRITICAL FORECAST VARIANCE: Period {{ period_id }} shows critical variance. Reasons: {{ variance_reasons | join(', ') }}",
                        "severity": "critical"
                    }
                },
                {
                    "id": "trigger_significant_alert",
                    "type": "notify",
                    "params": {
                        "channel": "email",
                        "recipient": "finance-team@company.com",
                        "subject": "Significant Forecast Variance Alert - Period {{ period_id }}",
                        "body": "Period {{ period_id }} shows significant forecast variance. Review required.",
                        "priority": "high"
                    }
                },
                {
                    "id": "log_normal_variance",
                    "type": "governance",
                    "params": {
                        "action": "log_event",
                        "event_type": "forecast_variance_normal",
                        "details": {
                            "period_id": "{{ period_id }}",
                            "variance_category": "{{ variance_category }}"
                        }
                    }
                }
            ],
            governance_config={
                "policy_pack_id": "saas_forecast_governance_v1",
                "trust_score_threshold": 0.70,
                "evidence_pack_required": True,
                "audit_logging": True
            },
            explainability_config={
                "enabled": True,
                "methods": ["SHAP", "feature_importance"],
                "explanation_storage": True
            },
            confidence_thresholds={
                "variance_detection": 0.70,
                "overall_confidence": 0.65
            },
            sample_data_schema={
                "period_id": "string",
                "forecasted_mrr": "float",
                "actual_mrr": "float",
                "forecasted_new_customers": "integer",
                "actual_new_customers": "integer",
                "forecasted_churn_rate": "float",
                "actual_churn_rate": "float",
                "seasonal_factor": "float",
                "market_conditions": "string"
            }
        )
        
        # Register SaaS templates
        self.templates[churn_risk_template.template_id] = churn_risk_template
        self.templates[forecast_variance_template.template_id] = forecast_variance_template
        
        self.logger.info("Created SaaS templates: Churn Risk Alert, Forecast Variance Detector")
    
    def _create_banking_templates(self):
        """Create Banking industry templates"""
        
        # Banking: Credit Scoring Check Template
        credit_scoring_template = TemplateConfig(
            template_id="banking_credit_scoring_check",
            template_name="Banking Credit Scoring Check",
            industry="Banking",
            template_type="credit_scoring",
            description="Automated credit scoring and approval/rejection workflow for loan applications",
            ml_models=[
                {
                    "model_id": "banking_credit_scorer_v4",
                    "model_name": "Banking Credit Scorer v4",
                    "model_type": "score",
                    "input_features": [
                        "credit_history_length",
                        "annual_income",
                        "debt_to_income_ratio",
                        "employment_status",
                        "existing_credit_accounts",
                        "payment_history_score",
                        "credit_utilization_ratio",
                        "loan_amount",
                        "loan_purpose"
                    ],
                    "output_features": ["credit_score", "risk_category"],
                    "confidence_threshold": 0.80,
                    "explainability_enabled": True
                },
                {
                    "model_id": "banking_income_verifier_v2",
                    "model_name": "Banking Income Verifier v2",
                    "model_type": "classify",
                    "input_features": [
                        "stated_income",
                        "employment_type",
                        "industry_sector",
                        "bank_statements_months",
                        "income_consistency_score"
                    ],
                    "output_features": ["income_verification_status"],
                    "confidence_threshold": 0.75,
                    "explainability_enabled": True
                }
            ],
            workflow_steps=[
                {
                    "id": "fetch_application_data",
                    "type": "query",
                    "params": {
                        "source": "loan_applications_db",
                        "query": """
                            SELECT 
                                application_id,
                                applicant_id,
                                credit_history_length,
                                annual_income,
                                debt_to_income_ratio,
                                employment_status,
                                existing_credit_accounts,
                                payment_history_score,
                                credit_utilization_ratio,
                                loan_amount,
                                loan_purpose,
                                stated_income,
                                employment_type,
                                industry_sector,
                                bank_statements_months,
                                income_consistency_score
                            FROM loan_applications 
                            WHERE application_id = {{ application_id }}
                        """
                    }
                },
                {
                    "id": "verify_income",
                    "type": "ml_classify",
                    "params": {
                        "model_id": "banking_income_verifier_v2",
                        "input_data_map": {
                            "stated_income": "application_data.stated_income",
                            "employment_type": "application_data.employment_type",
                            "industry_sector": "application_data.industry_sector",
                            "bank_statements_months": "application_data.bank_statements_months",
                            "income_consistency_score": "application_data.income_consistency_score"
                        },
                        "classes": ["verified", "needs_review", "rejected"],
                        "confidence_threshold": 0.75,
                        "explainability_enabled": True
                    }
                },
                {
                    "id": "calculate_credit_score",
                    "type": "ml_score",
                    "params": {
                        "model_id": "banking_credit_scorer_v4",
                        "input_data_map": {
                            "credit_history_length": "application_data.credit_history_length",
                            "annual_income": "application_data.annual_income",
                            "debt_to_income_ratio": "application_data.debt_to_income_ratio",
                            "employment_status": "application_data.employment_status",
                            "existing_credit_accounts": "application_data.existing_credit_accounts",
                            "payment_history_score": "application_data.payment_history_score",
                            "credit_utilization_ratio": "application_data.credit_utilization_ratio",
                            "loan_amount": "application_data.loan_amount",
                            "loan_purpose": "application_data.loan_purpose"
                        },
                        "confidence_threshold": 0.80,
                        "explainability_enabled": True
                    }
                },
                {
                    "id": "evaluate_application",
                    "type": "decision",
                    "params": {
                        "expression": "income_verification_status == 'verified' and credit_score >= 700",
                        "on_true": "approve_application",
                        "on_false": "check_conditional_approval"
                    }
                },
                {
                    "id": "check_conditional_approval",
                    "type": "decision",
                    "params": {
                        "expression": "income_verification_status == 'verified' and credit_score >= 600 and credit_score < 700",
                        "on_true": "conditional_approval",
                        "on_false": "reject_application"
                    }
                },
                {
                    "id": "approve_application",
                    "type": "notify",
                    "params": {
                        "channel": "database",
                        "table": "loan_decisions",
                        "data": {
                            "application_id": "{{ application_id }}",
                            "decision": "approved",
                            "credit_score": "{{ credit_score }}",
                            "decision_timestamp": "{{ current_timestamp }}"
                        }
                    }
                },
                {
                    "id": "conditional_approval",
                    "type": "notify",
                    "params": {
                        "channel": "human_review",
                        "queue": "loan_review_queue",
                        "data": {
                            "application_id": "{{ application_id }}",
                            "decision": "conditional_approval",
                            "credit_score": "{{ credit_score }}",
                            "review_reason": "Credit score in conditional range"
                        }
                    }
                },
                {
                    "id": "reject_application",
                    "type": "notify",
                    "params": {
                        "channel": "database",
                        "table": "loan_decisions",
                        "data": {
                            "application_id": "{{ application_id }}",
                            "decision": "rejected",
                            "credit_score": "{{ credit_score }}",
                            "rejection_reason": "Credit score below threshold or income verification failed"
                        }
                    }
                }
            ],
            governance_config={
                "policy_pack_id": "banking_credit_governance_v1",
                "trust_score_threshold": 0.80,
                "evidence_pack_required": True,
                "audit_logging": True,
                "regulatory_compliance": ["Basel III", "GDPR", "Fair Credit Reporting Act"],
                "bias_monitoring": True,
                "drift_monitoring": True
            },
            explainability_config={
                "enabled": True,
                "methods": ["SHAP", "LIME", "reason_codes"],
                "regulatory_explanation": True,
                "customer_facing_explanation": True,
                "explanation_storage": True
            },
            confidence_thresholds={
                "credit_scoring": 0.80,
                "income_verification": 0.75,
                "overall_confidence": 0.75
            },
            sample_data_schema={
                "application_id": "string",
                "applicant_id": "string",
                "credit_history_length": "integer",
                "annual_income": "float",
                "debt_to_income_ratio": "float",
                "employment_status": "string",
                "existing_credit_accounts": "integer",
                "payment_history_score": "float",
                "credit_utilization_ratio": "float",
                "loan_amount": "float",
                "loan_purpose": "string",
                "stated_income": "float",
                "employment_type": "string",
                "industry_sector": "string",
                "bank_statements_months": "integer",
                "income_consistency_score": "float"
            }
        )
        
        # Banking: Fraudulent Disbursal Detector Template
        fraud_disbursal_template = TemplateConfig(
            template_id="banking_fraudulent_disbursal_detector",
            template_name="Banking Fraudulent Disbursal Detector",
            industry="Banking",
            template_type="fraud_detection",
            description="Real-time fraud detection for loan disbursals and fund transfers",
            ml_models=[
                {
                    "model_id": "banking_fraud_detector_v5",
                    "model_name": "Banking Fraud Detector v5",
                    "model_type": "predict",
                    "input_features": [
                        "transaction_amount",
                        "recipient_account_age",
                        "sender_account_history",
                        "transaction_time",
                        "geographic_location",
                        "device_fingerprint",
                        "transaction_pattern_score",
                        "recipient_risk_score",
                        "velocity_check_score"
                    ],
                    "output_features": [
                        "fraud_probability",
                        "fraud_type",
                        "risk_indicators"
                    ],
                    "confidence_threshold": 0.85,
                    "explainability_enabled": True
                }
            ],
            workflow_steps=[
                {
                    "id": "fetch_transaction_data",
                    "type": "query",
                    "params": {
                        "source": "transactions_db",
                        "query": """
                            SELECT 
                                transaction_id,
                                transaction_amount,
                                recipient_account_age,
                                sender_account_history,
                                transaction_time,
                                geographic_location,
                                device_fingerprint,
                                transaction_pattern_score,
                                recipient_risk_score,
                                velocity_check_score
                            FROM pending_transactions 
                            WHERE transaction_id = {{ transaction_id }}
                        """
                    }
                },
                {
                    "id": "detect_fraud_risk",
                    "type": "ml_predict",
                    "params": {
                        "model_id": "banking_fraud_detector_v5",
                        "input_data_map": {
                            "transaction_amount": "transaction_data.transaction_amount",
                            "recipient_account_age": "transaction_data.recipient_account_age",
                            "sender_account_history": "transaction_data.sender_account_history",
                            "transaction_time": "transaction_data.transaction_time",
                            "geographic_location": "transaction_data.geographic_location",
                            "device_fingerprint": "transaction_data.device_fingerprint",
                            "transaction_pattern_score": "transaction_data.transaction_pattern_score",
                            "recipient_risk_score": "transaction_data.recipient_risk_score",
                            "velocity_check_score": "transaction_data.velocity_check_score"
                        },
                        "confidence_threshold": 0.85,
                        "explainability_enabled": True
                    }
                },
                {
                    "id": "evaluate_fraud_risk",
                    "type": "decision",
                    "params": {
                        "expression": "fraud_probability > 0.8",
                        "on_true": "block_transaction",
                        "on_false": "check_medium_risk"
                    }
                },
                {
                    "id": "check_medium_risk",
                    "type": "decision",
                    "params": {
                        "expression": "fraud_probability > 0.5",
                        "on_true": "flag_for_review",
                        "on_false": "approve_transaction"
                    }
                },
                {
                    "id": "block_transaction",
                    "type": "notify",
                    "params": {
                        "channel": "fraud_system",
                        "action": "block_transaction",
                        "data": {
                            "transaction_id": "{{ transaction_id }}",
                            "fraud_probability": "{{ fraud_probability }}",
                            "fraud_type": "{{ fraud_type }}",
                            "risk_indicators": "{{ risk_indicators }}"
                        }
                    }
                },
                {
                    "id": "flag_for_review",
                    "type": "notify",
                    "params": {
                        "channel": "fraud_review_queue",
                        "priority": "high",
                        "data": {
                            "transaction_id": "{{ transaction_id }}",
                            "fraud_probability": "{{ fraud_probability }}",
                            "review_required": True
                        }
                    }
                },
                {
                    "id": "approve_transaction",
                    "type": "notify",
                    "params": {
                        "channel": "transaction_processor",
                        "action": "approve_transaction",
                        "data": {
                            "transaction_id": "{{ transaction_id }}",
                            "fraud_probability": "{{ fraud_probability }}"
                        }
                    }
                }
            ],
            governance_config={
                "policy_pack_id": "banking_fraud_governance_v1",
                "trust_score_threshold": 0.85,
                "evidence_pack_required": True,
                "audit_logging": True,
                "regulatory_compliance": ["AML", "BSA", "OFAC"],
                "real_time_monitoring": True
            },
            explainability_config={
                "enabled": True,
                "methods": ["SHAP", "reason_codes"],
                "real_time_explanation": True,
                "explanation_storage": True
            },
            confidence_thresholds={
                "fraud_detection": 0.85,
                "overall_confidence": 0.80
            },
            sample_data_schema={
                "transaction_id": "string",
                "transaction_amount": "float",
                "recipient_account_age": "integer",
                "sender_account_history": "float",
                "transaction_time": "datetime",
                "geographic_location": "string",
                "device_fingerprint": "string",
                "transaction_pattern_score": "float",
                "recipient_risk_score": "float",
                "velocity_check_score": "float"
            }
        )
        
        # Register Banking templates
        self.templates[credit_scoring_template.template_id] = credit_scoring_template
        self.templates[fraud_disbursal_template.template_id] = fraud_disbursal_template
        
        self.logger.info("Created Banking templates: Credit Scoring Check, Fraudulent Disbursal Detector")
    
    def _create_insurance_templates(self):
        """Create Insurance industry templates"""
        
        # Insurance: Claim Fraud Anomaly Template
        claim_fraud_template = TemplateConfig(
            template_id="insurance_claim_fraud_anomaly",
            template_name="Insurance Claim Fraud Anomaly",
            industry="Insurance",
            template_type="fraud_detection",
            description="Automated fraud detection for insurance claims using anomaly detection",
            ml_models=[
                {
                    "model_id": "insurance_fraud_detector_v3",
                    "model_name": "Insurance Fraud Detector v3",
                    "model_type": "predict",
                    "input_features": [
                        "claim_amount",
                        "policy_age",
                        "claimant_history",
                        "incident_location",
                        "incident_time",
                        "claim_description_sentiment",
                        "supporting_documents_count",
                        "witness_statements_count",
                        "medical_provider_history",
                        "repair_shop_history"
                    ],
                    "output_features": [
                        "fraud_probability",
                        "fraud_indicators",
                        "investigation_priority"
                    ],
                    "confidence_threshold": 0.75,
                    "explainability_enabled": True
                }
            ],
            workflow_steps=[
                {
                    "id": "fetch_claim_data",
                    "type": "query",
                    "params": {
                        "source": "claims_db",
                        "query": """
                            SELECT 
                                claim_id,
                                claim_amount,
                                policy_age,
                                claimant_history,
                                incident_location,
                                incident_time,
                                claim_description_sentiment,
                                supporting_documents_count,
                                witness_statements_count,
                                medical_provider_history,
                                repair_shop_history
                            FROM insurance_claims 
                            WHERE claim_id = {{ claim_id }}
                        """
                    }
                },
                {
                    "id": "detect_claim_fraud",
                    "type": "ml_predict",
                    "params": {
                        "model_id": "insurance_fraud_detector_v3",
                        "input_data_map": {
                            "claim_amount": "claim_data.claim_amount",
                            "policy_age": "claim_data.policy_age",
                            "claimant_history": "claim_data.claimant_history",
                            "incident_location": "claim_data.incident_location",
                            "incident_time": "claim_data.incident_time",
                            "claim_description_sentiment": "claim_data.claim_description_sentiment",
                            "supporting_documents_count": "claim_data.supporting_documents_count",
                            "witness_statements_count": "claim_data.witness_statements_count",
                            "medical_provider_history": "claim_data.medical_provider_history",
                            "repair_shop_history": "claim_data.repair_shop_history"
                        },
                        "confidence_threshold": 0.75,
                        "explainability_enabled": True
                    }
                },
                {
                    "id": "evaluate_fraud_risk",
                    "type": "decision",
                    "params": {
                        "expression": "fraud_probability > 0.8",
                        "on_true": "flag_for_investigation",
                        "on_false": "check_medium_risk"
                    }
                },
                {
                    "id": "check_medium_risk",
                    "type": "decision",
                    "params": {
                        "expression": "fraud_probability > 0.5",
                        "on_true": "flag_for_review",
                        "on_false": "approve_claim"
                    }
                },
                {
                    "id": "flag_for_investigation",
                    "type": "notify",
                    "params": {
                        "channel": "siu_queue",
                        "priority": "high",
                        "data": {
                            "claim_id": "{{ claim_id }}",
                            "fraud_probability": "{{ fraud_probability }}",
                            "fraud_indicators": "{{ fraud_indicators }}",
                            "investigation_priority": "{{ investigation_priority }}"
                        }
                    }
                },
                {
                    "id": "flag_for_review",
                    "type": "notify",
                    "params": {
                        "channel": "claims_review_queue",
                        "priority": "medium",
                        "data": {
                            "claim_id": "{{ claim_id }}",
                            "fraud_probability": "{{ fraud_probability }}",
                            "review_required": True
                        }
                    }
                },
                {
                    "id": "approve_claim",
                    "type": "notify",
                    "params": {
                        "channel": "claims_processor",
                        "action": "approve_claim",
                        "data": {
                            "claim_id": "{{ claim_id }}",
                            "fraud_probability": "{{ fraud_probability }}"
                        }
                    }
                }
            ],
            governance_config={
                "policy_pack_id": "insurance_fraud_governance_v1",
                "trust_score_threshold": 0.75,
                "evidence_pack_required": True,
                "audit_logging": True,
                "regulatory_compliance": ["NAIC", "State Insurance Regulations"]
            },
            explainability_config={
                "enabled": True,
                "methods": ["SHAP", "reason_codes"],
                "investigation_support": True,
                "explanation_storage": True
            },
            confidence_thresholds={
                "fraud_detection": 0.75,
                "overall_confidence": 0.70
            },
            sample_data_schema={
                "claim_id": "string",
                "claim_amount": "float",
                "policy_age": "integer",
                "claimant_history": "float",
                "incident_location": "string",
                "incident_time": "datetime",
                "claim_description_sentiment": "float",
                "supporting_documents_count": "integer",
                "witness_statements_count": "integer",
                "medical_provider_history": "float",
                "repair_shop_history": "float"
            }
        )
        
        # Insurance: Policy Lapse Predictor Template
        policy_lapse_template = TemplateConfig(
            template_id="insurance_policy_lapse_predictor",
            template_name="Insurance Policy Lapse Predictor",
            industry="Insurance",
            template_type="lapse_prediction",
            description="Predicts policy lapses and triggers retention actions",
            ml_models=[
                {
                    "model_id": "insurance_lapse_predictor_v2",
                    "model_name": "Insurance Lapse Predictor v2",
                    "model_type": "predict",
                    "input_features": [
                        "policy_age",
                        "premium_amount",
                        "payment_history",
                        "claims_history",
                        "customer_age",
                        "customer_income_bracket",
                        "policy_type",
                        "coverage_amount",
                        "last_payment_days_ago",
                        "customer_satisfaction_score"
                    ],
                    "output_features": [
                        "lapse_probability",
                        "lapse_timeframe",
                        "retention_recommendations"
                    ],
                    "confidence_threshold": 0.70,
                    "explainability_enabled": True
                }
            ],
            workflow_steps=[
                {
                    "id": "fetch_policy_data",
                    "type": "query",
                    "params": {
                        "source": "policies_db",
                        "query": """
                            SELECT 
                                policy_id,
                                policy_age,
                                premium_amount,
                                payment_history,
                                claims_history,
                                customer_age,
                                customer_income_bracket,
                                policy_type,
                                coverage_amount,
                                last_payment_days_ago,
                                customer_satisfaction_score
                            FROM insurance_policies 
                            WHERE policy_id = {{ policy_id }}
                        """
                    }
                },
                {
                    "id": "predict_policy_lapse",
                    "type": "ml_predict",
                    "params": {
                        "model_id": "insurance_lapse_predictor_v2",
                        "input_data_map": {
                            "policy_age": "policy_data.policy_age",
                            "premium_amount": "policy_data.premium_amount",
                            "payment_history": "policy_data.payment_history",
                            "claims_history": "policy_data.claims_history",
                            "customer_age": "policy_data.customer_age",
                            "customer_income_bracket": "policy_data.customer_income_bracket",
                            "policy_type": "policy_data.policy_type",
                            "coverage_amount": "policy_data.coverage_amount",
                            "last_payment_days_ago": "policy_data.last_payment_days_ago",
                            "customer_satisfaction_score": "policy_data.customer_satisfaction_score"
                        },
                        "confidence_threshold": 0.70,
                        "explainability_enabled": True
                    }
                },
                {
                    "id": "evaluate_lapse_risk",
                    "type": "decision",
                    "params": {
                        "expression": "lapse_probability > 0.7",
                        "on_true": "trigger_high_risk_retention",
                        "on_false": "check_medium_risk"
                    }
                },
                {
                    "id": "check_medium_risk",
                    "type": "decision",
                    "params": {
                        "expression": "lapse_probability > 0.4",
                        "on_true": "trigger_medium_risk_retention",
                        "on_false": "log_low_risk"
                    }
                },
                {
                    "id": "trigger_high_risk_retention",
                    "type": "notify",
                    "params": {
                        "channel": "retention_team",
                        "priority": "high",
                        "data": {
                            "policy_id": "{{ policy_id }}",
                            "lapse_probability": "{{ lapse_probability }}",
                            "lapse_timeframe": "{{ lapse_timeframe }}",
                            "retention_recommendations": "{{ retention_recommendations }}"
                        }
                    }
                },
                {
                    "id": "trigger_medium_risk_retention",
                    "type": "notify",
                    "params": {
                        "channel": "email",
                        "recipient": "customer-retention@insurance.com",
                        "subject": "Policy Lapse Risk Alert - {{ policy_id }}",
                        "body": "Policy {{ policy_id }} has {{ lapse_probability | round(2) }}% lapse probability. Consider proactive retention.",
                        "priority": "medium"
                    }
                },
                {
                    "id": "log_low_risk",
                    "type": "governance",
                    "params": {
                        "action": "log_event",
                        "event_type": "policy_lapse_low_risk",
                        "details": {
                            "policy_id": "{{ policy_id }}",
                            "lapse_probability": "{{ lapse_probability }}"
                        }
                    }
                }
            ],
            governance_config={
                "policy_pack_id": "insurance_lapse_governance_v1",
                "trust_score_threshold": 0.70,
                "evidence_pack_required": True,
                "audit_logging": True
            },
            explainability_config={
                "enabled": True,
                "methods": ["SHAP", "reason_codes"],
                "retention_insights": True,
                "explanation_storage": True
            },
            confidence_thresholds={
                "lapse_prediction": 0.70,
                "overall_confidence": 0.65
            },
            sample_data_schema={
                "policy_id": "string",
                "policy_age": "integer",
                "premium_amount": "float",
                "payment_history": "float",
                "claims_history": "float",
                "customer_age": "integer",
                "customer_income_bracket": "string",
                "policy_type": "string",
                "coverage_amount": "float",
                "last_payment_days_ago": "integer",
                "customer_satisfaction_score": "float"
            }
        )
        
        # Register Insurance templates
        self.templates[claim_fraud_template.template_id] = claim_fraud_template
        self.templates[policy_lapse_template.template_id] = policy_lapse_template
        
        self.logger.info("Created Insurance templates: Claim Fraud Anomaly, Policy Lapse Predictor")
    
    def _create_ecommerce_templates(self):
        """Create E-commerce industry templates"""
        
        # E-commerce: Fraud Scoring at Checkout Template
        checkout_fraud_template = TemplateConfig(
            template_id="ecommerce_checkout_fraud_scoring",
            template_name="E-commerce Checkout Fraud Scoring",
            industry="E-commerce",
            template_type="fraud_detection",
            description="Real-time fraud scoring for e-commerce checkout transactions",
            ml_models=[
                {
                    "model_id": "ecommerce_fraud_scorer_v4",
                    "model_name": "E-commerce Fraud Scorer v4",
                    "model_type": "score",
                    "input_features": [
                        "transaction_amount",
                        "customer_account_age",
                        "shipping_billing_match",
                        "payment_method",
                        "device_fingerprint",
                        "ip_geolocation",
                        "order_velocity",
                        "product_category_risk",
                        "customer_purchase_history",
                        "time_of_purchase"
                    ],
                    "output_features": ["fraud_score", "risk_factors"],
                    "confidence_threshold": 0.75,
                    "explainability_enabled": True
                }
            ],
            workflow_steps=[
                {
                    "id": "fetch_checkout_data",
                    "type": "query",
                    "params": {
                        "source": "checkout_db",
                        "query": """
                            SELECT 
                                order_id,
                                transaction_amount,
                                customer_account_age,
                                shipping_billing_match,
                                payment_method,
                                device_fingerprint,
                                ip_geolocation,
                                order_velocity,
                                product_category_risk,
                                customer_purchase_history,
                                time_of_purchase
                            FROM checkout_transactions 
                            WHERE order_id = {{ order_id }}
                        """
                    }
                },
                {
                    "id": "calculate_fraud_score",
                    "type": "ml_score",
                    "params": {
                        "model_id": "ecommerce_fraud_scorer_v4",
                        "input_data_map": {
                            "transaction_amount": "checkout_data.transaction_amount",
                            "customer_account_age": "checkout_data.customer_account_age",
                            "shipping_billing_match": "checkout_data.shipping_billing_match",
                            "payment_method": "checkout_data.payment_method",
                            "device_fingerprint": "checkout_data.device_fingerprint",
                            "ip_geolocation": "checkout_data.ip_geolocation",
                            "order_velocity": "checkout_data.order_velocity",
                            "product_category_risk": "checkout_data.product_category_risk",
                            "customer_purchase_history": "checkout_data.customer_purchase_history",
                            "time_of_purchase": "checkout_data.time_of_purchase"
                        },
                        "confidence_threshold": 0.75,
                        "explainability_enabled": True
                    }
                },
                {
                    "id": "evaluate_fraud_score",
                    "type": "decision",
                    "params": {
                        "expression": "fraud_score > 80",
                        "on_true": "block_transaction",
                        "on_false": "check_medium_risk"
                    }
                },
                {
                    "id": "check_medium_risk",
                    "type": "decision",
                    "params": {
                        "expression": "fraud_score > 50",
                        "on_true": "require_additional_verification",
                        "on_false": "approve_transaction"
                    }
                },
                {
                    "id": "block_transaction",
                    "type": "notify",
                    "params": {
                        "channel": "fraud_prevention",
                        "action": "block_order",
                        "data": {
                            "order_id": "{{ order_id }}",
                            "fraud_score": "{{ fraud_score }}",
                            "risk_factors": "{{ risk_factors }}"
                        }
                    }
                },
                {
                    "id": "require_additional_verification",
                    "type": "notify",
                    "params": {
                        "channel": "verification_service",
                        "action": "request_verification",
                        "data": {
                            "order_id": "{{ order_id }}",
                            "fraud_score": "{{ fraud_score }}",
                            "verification_type": "2FA"
                        }
                    }
                },
                {
                    "id": "approve_transaction",
                    "type": "notify",
                    "params": {
                        "channel": "order_processor",
                        "action": "approve_order",
                        "data": {
                            "order_id": "{{ order_id }}",
                            "fraud_score": "{{ fraud_score }}"
                        }
                    }
                }
            ],
            governance_config={
                "policy_pack_id": "ecommerce_fraud_governance_v1",
                "trust_score_threshold": 0.75,
                "evidence_pack_required": True,
                "audit_logging": True,
                "real_time_processing": True
            },
            explainability_config={
                "enabled": True,
                "methods": ["SHAP", "reason_codes"],
                "customer_facing_explanation": False,
                "explanation_storage": True
            },
            confidence_thresholds={
                "fraud_scoring": 0.75,
                "overall_confidence": 0.70
            },
            sample_data_schema={
                "order_id": "string",
                "transaction_amount": "float",
                "customer_account_age": "integer",
                "shipping_billing_match": "boolean",
                "payment_method": "string",
                "device_fingerprint": "string",
                "ip_geolocation": "string",
                "order_velocity": "float",
                "product_category_risk": "float",
                "customer_purchase_history": "float",
                "time_of_purchase": "datetime"
            }
        )
        
        # E-commerce: Refund Delay Predictor Template
        refund_delay_template = TemplateConfig(
            template_id="ecommerce_refund_delay_predictor",
            template_name="E-commerce Refund Delay Predictor",
            industry="E-commerce",
            template_type="delay_prediction",
            description="Predicts potential delays in refund processing and optimizes workflow",
            ml_models=[
                {
                    "model_id": "ecommerce_refund_delay_predictor_v2",
                    "model_name": "E-commerce Refund Delay Predictor v2",
                    "model_type": "predict",
                    "input_features": [
                        "refund_amount",
                        "product_category",
                        "refund_reason",
                        "customer_tier",
                        "order_age",
                        "payment_method",
                        "return_condition",
                        "processing_queue_length",
                        "staff_availability",
                        "complexity_score"
                    ],
                    "output_features": [
                        "delay_probability",
                        "estimated_processing_days",
                        "bottleneck_factors"
                    ],
                    "confidence_threshold": 0.70,
                    "explainability_enabled": True
                }
            ],
            workflow_steps=[
                {
                    "id": "fetch_refund_data",
                    "type": "query",
                    "params": {
                        "source": "refunds_db",
                        "query": """
                            SELECT 
                                refund_id,
                                refund_amount,
                                product_category,
                                refund_reason,
                                customer_tier,
                                order_age,
                                payment_method,
                                return_condition,
                                processing_queue_length,
                                staff_availability,
                                complexity_score
                            FROM refund_requests 
                            WHERE refund_id = {{ refund_id }}
                        """
                    }
                },
                {
                    "id": "predict_refund_delay",
                    "type": "ml_predict",
                    "params": {
                        "model_id": "ecommerce_refund_delay_predictor_v2",
                        "input_data_map": {
                            "refund_amount": "refund_data.refund_amount",
                            "product_category": "refund_data.product_category",
                            "refund_reason": "refund_data.refund_reason",
                            "customer_tier": "refund_data.customer_tier",
                            "order_age": "refund_data.order_age",
                            "payment_method": "refund_data.payment_method",
                            "return_condition": "refund_data.return_condition",
                            "processing_queue_length": "refund_data.processing_queue_length",
                            "staff_availability": "refund_data.staff_availability",
                            "complexity_score": "refund_data.complexity_score"
                        },
                        "confidence_threshold": 0.70,
                        "explainability_enabled": True
                    }
                },
                {
                    "id": "evaluate_delay_risk",
                    "type": "decision",
                    "params": {
                        "expression": "delay_probability > 0.7",
                        "on_true": "prioritize_refund",
                        "on_false": "check_medium_risk"
                    }
                },
                {
                    "id": "check_medium_risk",
                    "type": "decision",
                    "params": {
                        "expression": "delay_probability > 0.4",
                        "on_true": "flag_for_monitoring",
                        "on_false": "process_normally"
                    }
                },
                {
                    "id": "prioritize_refund",
                    "type": "notify",
                    "params": {
                        "channel": "priority_queue",
                        "action": "prioritize_refund",
                        "data": {
                            "refund_id": "{{ refund_id }}",
                            "delay_probability": "{{ delay_probability }}",
                            "estimated_processing_days": "{{ estimated_processing_days }}",
                            "bottleneck_factors": "{{ bottleneck_factors }}"
                        }
                    }
                },
                {
                    "id": "flag_for_monitoring",
                    "type": "notify",
                    "params": {
                        "channel": "monitoring_queue",
                        "action": "monitor_refund",
                        "data": {
                            "refund_id": "{{ refund_id }}",
                            "delay_probability": "{{ delay_probability }}"
                        }
                    }
                },
                {
                    "id": "process_normally",
                    "type": "notify",
                    "params": {
                        "channel": "standard_queue",
                        "action": "process_refund",
                        "data": {
                            "refund_id": "{{ refund_id }}",
                            "estimated_processing_days": "{{ estimated_processing_days }}"
                        }
                    }
                }
            ],
            governance_config={
                "policy_pack_id": "ecommerce_refund_governance_v1",
                "trust_score_threshold": 0.70,
                "evidence_pack_required": True,
                "audit_logging": True
            },
            explainability_config={
                "enabled": True,
                "methods": ["SHAP", "reason_codes"],
                "operational_insights": True,
                "explanation_storage": True
            },
            confidence_thresholds={
                "delay_prediction": 0.70,
                "overall_confidence": 0.65
            },
            sample_data_schema={
                "refund_id": "string",
                "refund_amount": "float",
                "product_category": "string",
                "refund_reason": "string",
                "customer_tier": "string",
                "order_age": "integer",
                "payment_method": "string",
                "return_condition": "string",
                "processing_queue_length": "integer",
                "staff_availability": "float",
                "complexity_score": "float"
            }
        )
        
        # Register E-commerce templates
        self.templates[checkout_fraud_template.template_id] = checkout_fraud_template
        self.templates[refund_delay_template.template_id] = refund_delay_template
        
        self.logger.info("Created E-commerce templates: Checkout Fraud Scoring, Refund Delay Predictor")
    
    def _create_financial_services_templates(self):
        """Create Financial Services industry templates"""
        
        # Financial Services: Liquidity Risk Early Warning Template
        liquidity_risk_template = TemplateConfig(
            template_id="fs_liquidity_risk_early_warning",
            template_name="Financial Services Liquidity Risk Early Warning",
            industry="Financial Services",
            template_type="risk_management",
            description="Early warning system for liquidity risk in financial institutions",
            ml_models=[
                {
                    "model_id": "fs_liquidity_risk_detector_v3",
                    "model_name": "FS Liquidity Risk Detector v3",
                    "model_type": "predict",
                    "input_features": [
                        "cash_reserves_ratio",
                        "deposit_withdrawal_velocity",
                        "loan_to_deposit_ratio",
                        "interbank_borrowing_rate",
                        "market_volatility_index",
                        "regulatory_capital_ratio",
                        "stress_test_results",
                        "funding_concentration",
                        "maturity_mismatch_ratio",
                        "liquidity_coverage_ratio"
                    ],
                    "output_features": [
                        "liquidity_risk_score",
                        "risk_level",
                        "time_to_critical"
                    ],
                    "confidence_threshold": 0.80,
                    "explainability_enabled": True
                }
            ],
            workflow_steps=[
                {
                    "id": "fetch_liquidity_metrics",
                    "type": "query",
                    "params": {
                        "source": "risk_management_db",
                        "query": """
                            SELECT 
                                institution_id,
                                cash_reserves_ratio,
                                deposit_withdrawal_velocity,
                                loan_to_deposit_ratio,
                                interbank_borrowing_rate,
                                market_volatility_index,
                                regulatory_capital_ratio,
                                stress_test_results,
                                funding_concentration,
                                maturity_mismatch_ratio,
                                liquidity_coverage_ratio
                            FROM liquidity_metrics 
                            WHERE institution_id = {{ institution_id }}
                            AND date = CURRENT_DATE
                        """
                    }
                },
                {
                    "id": "assess_liquidity_risk",
                    "type": "ml_predict",
                    "params": {
                        "model_id": "fs_liquidity_risk_detector_v3",
                        "input_data_map": {
                            "cash_reserves_ratio": "liquidity_metrics.cash_reserves_ratio",
                            "deposit_withdrawal_velocity": "liquidity_metrics.deposit_withdrawal_velocity",
                            "loan_to_deposit_ratio": "liquidity_metrics.loan_to_deposit_ratio",
                            "interbank_borrowing_rate": "liquidity_metrics.interbank_borrowing_rate",
                            "market_volatility_index": "liquidity_metrics.market_volatility_index",
                            "regulatory_capital_ratio": "liquidity_metrics.regulatory_capital_ratio",
                            "stress_test_results": "liquidity_metrics.stress_test_results",
                            "funding_concentration": "liquidity_metrics.funding_concentration",
                            "maturity_mismatch_ratio": "liquidity_metrics.maturity_mismatch_ratio",
                            "liquidity_coverage_ratio": "liquidity_metrics.liquidity_coverage_ratio"
                        },
                        "confidence_threshold": 0.80,
                        "explainability_enabled": True
                    }
                },
                {
                    "id": "evaluate_risk_level",
                    "type": "decision",
                    "params": {
                        "expression": "risk_level == 'critical'",
                        "on_true": "trigger_critical_alert",
                        "on_false": "check_high_risk"
                    }
                },
                {
                    "id": "check_high_risk",
                    "type": "decision",
                    "params": {
                        "expression": "risk_level == 'high'",
                        "on_true": "trigger_high_risk_alert",
                        "on_false": "check_medium_risk"
                    }
                },
                {
                    "id": "check_medium_risk",
                    "type": "decision",
                    "params": {
                        "expression": "risk_level == 'medium'",
                        "on_true": "trigger_medium_risk_alert",
                        "on_false": "log_normal_risk"
                    }
                },
                {
                    "id": "trigger_critical_alert",
                    "type": "notify",
                    "params": {
                        "channel": "emergency_response",
                        "priority": "critical",
                        "data": {
                            "institution_id": "{{ institution_id }}",
                            "liquidity_risk_score": "{{ liquidity_risk_score }}",
                            "risk_level": "{{ risk_level }}",
                            "time_to_critical": "{{ time_to_critical }}",
                            "immediate_action_required": True
                        }
                    }
                },
                {
                    "id": "trigger_high_risk_alert",
                    "type": "notify",
                    "params": {
                        "channel": "risk_management_team",
                        "priority": "high",
                        "data": {
                            "institution_id": "{{ institution_id }}",
                            "liquidity_risk_score": "{{ liquidity_risk_score }}",
                            "risk_level": "{{ risk_level }}"
                        }
                    }
                },
                {
                    "id": "trigger_medium_risk_alert",
                    "type": "notify",
                    "params": {
                        "channel": "email",
                        "recipient": "risk-monitoring@financialservices.com",
                        "subject": "Medium Liquidity Risk Alert - {{ institution_id }}",
                        "body": "Institution {{ institution_id }} shows medium liquidity risk. Score: {{ liquidity_risk_score }}",
                        "priority": "medium"
                    }
                },
                {
                    "id": "log_normal_risk",
                    "type": "governance",
                    "params": {
                        "action": "log_event",
                        "event_type": "liquidity_risk_normal",
                        "details": {
                            "institution_id": "{{ institution_id }}",
                            "liquidity_risk_score": "{{ liquidity_risk_score }}"
                        }
                    }
                }
            ],
            governance_config={
                "policy_pack_id": "fs_liquidity_governance_v1",
                "trust_score_threshold": 0.80,
                "evidence_pack_required": True,
                "audit_logging": True,
                "regulatory_compliance": ["Basel III", "Dodd-Frank", "MiFID II"],
                "real_time_monitoring": True
            },
            explainability_config={
                "enabled": True,
                "methods": ["SHAP", "LIME", "reason_codes"],
                "regulatory_explanation": True,
                "risk_factor_analysis": True,
                "explanation_storage": True
            },
            confidence_thresholds={
                "liquidity_risk_assessment": 0.80,
                "overall_confidence": 0.75
            },
            sample_data_schema={
                "institution_id": "string",
                "cash_reserves_ratio": "float",
                "deposit_withdrawal_velocity": "float",
                "loan_to_deposit_ratio": "float",
                "interbank_borrowing_rate": "float",
                "market_volatility_index": "float",
                "regulatory_capital_ratio": "float",
                "stress_test_results": "float",
                "funding_concentration": "float",
                "maturity_mismatch_ratio": "float",
                "liquidity_coverage_ratio": "float"
            }
        )
        
        # Financial Services: MiFID/Reg Reporting with Anomaly Detection Template
        mifid_reporting_template = TemplateConfig(
            template_id="fs_mifid_reporting_anomaly_detection",
            template_name="Financial Services MiFID/Reg Reporting with Anomaly Detection",
            industry="Financial Services",
            template_type="regulatory_reporting",
            description="Automated regulatory reporting with anomaly detection for MiFID and other regulations",
            ml_models=[
                {
                    "model_id": "fs_reporting_anomaly_detector_v2",
                    "model_name": "FS Reporting Anomaly Detector v2",
                    "model_type": "classify",
                    "input_features": [
                        "transaction_volume",
                        "transaction_value",
                        "client_classification",
                        "instrument_type",
                        "execution_venue",
                        "time_of_trade",
                        "price_deviation",
                        "volume_deviation",
                        "counterparty_risk_score",
                        "regulatory_flags"
                    ],
                    "output_features": [
                        "anomaly_category",
                        "reporting_requirement",
                        "compliance_risk_level"
                    ],
                    "confidence_threshold": 0.75,
                    "explainability_enabled": True
                }
            ],
            workflow_steps=[
                {
                    "id": "fetch_trading_data",
                    "type": "query",
                    "params": {
                        "source": "trading_db",
                        "query": """
                            SELECT 
                                trade_id,
                                transaction_volume,
                                transaction_value,
                                client_classification,
                                instrument_type,
                                execution_venue,
                                time_of_trade,
                                price_deviation,
                                volume_deviation,
                                counterparty_risk_score,
                                regulatory_flags
                            FROM trading_transactions 
                            WHERE trade_date = {{ trade_date }}
                            AND regulatory_reporting_required = true
                        """
                    }
                },
                {
                    "id": "detect_reporting_anomalies",
                    "type": "ml_classify",
                    "params": {
                        "model_id": "fs_reporting_anomaly_detector_v2",
                        "input_data_map": {
                            "transaction_volume": "trading_data.transaction_volume",
                            "transaction_value": "trading_data.transaction_value",
                            "client_classification": "trading_data.client_classification",
                            "instrument_type": "trading_data.instrument_type",
                            "execution_venue": "trading_data.execution_venue",
                            "time_of_trade": "trading_data.time_of_trade",
                            "price_deviation": "trading_data.price_deviation",
                            "volume_deviation": "trading_data.volume_deviation",
                            "counterparty_risk_score": "trading_data.counterparty_risk_score",
                            "regulatory_flags": "trading_data.regulatory_flags"
                        },
                        "classes": ["normal_reporting", "anomaly_detected", "high_risk_anomaly"],
                        "confidence_threshold": 0.75,
                        "explainability_enabled": True
                    }
                },
                {
                    "id": "evaluate_anomaly_level",
                    "type": "decision",
                    "params": {
                        "expression": "anomaly_category == 'high_risk_anomaly'",
                        "on_true": "flag_for_investigation",
                        "on_false": "check_standard_anomaly"
                    }
                },
                {
                    "id": "check_standard_anomaly",
                    "type": "decision",
                    "params": {
                        "expression": "anomaly_category == 'anomaly_detected'",
                        "on_true": "flag_for_review",
                        "on_false": "process_normal_reporting"
                    }
                },
                {
                    "id": "flag_for_investigation",
                    "type": "notify",
                    "params": {
                        "channel": "compliance_investigation",
                        "priority": "high",
                        "data": {
                            "trade_id": "{{ trade_id }}",
                            "anomaly_category": "{{ anomaly_category }}",
                            "reporting_requirement": "{{ reporting_requirement }}",
                            "compliance_risk_level": "{{ compliance_risk_level }}"
                        }
                    }
                },
                {
                    "id": "flag_for_review",
                    "type": "notify",
                    "params": {
                        "channel": "compliance_review",
                        "priority": "medium",
                        "data": {
                            "trade_id": "{{ trade_id }}",
                            "anomaly_category": "{{ anomaly_category }}"
                        }
                    }
                },
                {
                    "id": "process_normal_reporting",
                    "type": "notify",
                    "params": {
                        "channel": "regulatory_reporting_system",
                        "action": "generate_report",
                        "data": {
                            "trade_id": "{{ trade_id }}",
                            "reporting_requirement": "{{ reporting_requirement }}"
                        }
                    }
                }
            ],
            governance_config={
                "policy_pack_id": "fs_mifid_governance_v1",
                "trust_score_threshold": 0.75,
                "evidence_pack_required": True,
                "audit_logging": True,
                "regulatory_compliance": ["MiFID II", "EMIR", "MAR"],
                "automated_reporting": True
            },
            explainability_config={
                "enabled": True,
                "methods": ["SHAP", "reason_codes"],
                "regulatory_explanation": True,
                "compliance_documentation": True,
                "explanation_storage": True
            },
            confidence_thresholds={
                "anomaly_detection": 0.75,
                "overall_confidence": 0.70
            },
            sample_data_schema={
                "trade_id": "string",
                "transaction_volume": "float",
                "transaction_value": "float",
                "client_classification": "string",
                "instrument_type": "string",
                "execution_venue": "string",
                "time_of_trade": "datetime",
                "price_deviation": "float",
                "volume_deviation": "float",
                "counterparty_risk_score": "float",
                "regulatory_flags": "string"
            }
        )
        
        # Register Financial Services templates
        self.templates[liquidity_risk_template.template_id] = liquidity_risk_template
        self.templates[mifid_reporting_template.template_id] = mifid_reporting_template
        
        self.logger.info("Created Financial Services templates: Liquidity Risk Early Warning, MiFID/Reg Reporting with Anomaly Detection")
    
    def get_template(self, template_id: str) -> Optional[TemplateConfig]:
        """Get a template by ID"""
        return self.templates.get(template_id)
    
    def get_templates_by_industry(self, industry: str) -> List[TemplateConfig]:
        """Get all templates for a specific industry"""
        return [template for template in self.templates.values() if template.industry == industry]
    
    def get_templates_by_type(self, template_type: str) -> List[TemplateConfig]:
        """Get all templates of a specific type"""
        return [template for template in self.templates.values() if template.template_type == template_type]
    
    def list_all_templates(self) -> List[TemplateConfig]:
        """Get all available templates"""
        return list(self.templates.values())
    
    def generate_workflow_yaml(self, template_id: str, custom_params: Optional[Dict[str, Any]] = None) -> str:
        """Generate a YAML workflow from a template"""
        template = self.get_template(template_id)
        if not template:
            raise ValueError(f"Template {template_id} not found")
        
        workflow = {
            "workflow_id": f"{template_id}_workflow_{uuid.uuid4().hex[:8]}",
            "name": f"{template.template_name} Workflow",
            "module": template.industry.lower(),
            "automation_type": "RBIA",
            "version": template.version,
            "governance": template.governance_config,
            "metadata": {
                "description": template.description,
                "template_id": template.template_id,
                "industry": template.industry,
                "template_type": template.template_type,
                **template.metadata
            },
            "steps": template.workflow_steps
        }
        
        # Apply custom parameters if provided
        if custom_params:
            # This would merge custom parameters into the workflow
            # Implementation depends on specific customization requirements
            pass
        
        return yaml.dump(workflow, default_flow_style=False, sort_keys=False)
    
    def export_template(self, template_id: str) -> Dict[str, Any]:
        """Export a template configuration"""
        template = self.get_template(template_id)
        if not template:
            raise ValueError(f"Template {template_id} not found")
        
        return {
            "template_id": template.template_id,
            "template_name": template.template_name,
            "industry": template.industry,
            "template_type": template.template_type,
            "description": template.description,
            "ml_models": template.ml_models,
            "workflow_steps": template.workflow_steps,
            "governance_config": template.governance_config,
            "explainability_config": template.explainability_config,
            "confidence_thresholds": template.confidence_thresholds,
            "sample_data_schema": template.sample_data_schema,
            "metadata": template.metadata,
            "version": template.version,
            "created_at": template.created_at.isoformat()
        }
    
    def get_industry_summary(self) -> Dict[str, Any]:
        """Get summary of templates by industry"""
        summary = {}
        
        for template in self.templates.values():
            industry = template.industry
            if industry not in summary:
                summary[industry] = {
                    "template_count": 0,
                    "template_types": set(),
                    "templates": []
                }
            
            summary[industry]["template_count"] += 1
            summary[industry]["template_types"].add(template.template_type)
            summary[industry]["templates"].append({
                "template_id": template.template_id,
                "template_name": template.template_name,
                "template_type": template.template_type
            })
        
        # Convert sets to lists for JSON serialization
        for industry_data in summary.values():
            industry_data["template_types"] = list(industry_data["template_types"])
        
        return summary
