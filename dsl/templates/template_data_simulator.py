"""
Sample Data Simulators for Templates
Task 4.2.27: Create sample data simulators for templates
"""

import json
import uuid
import random
import asyncio
from typing import Dict, List, Any, Optional, Union, Generator
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import logging
import numpy as np
from enum import Enum

from .industry_template_registry import IndustryTemplateRegistry, TemplateConfig

logger = logging.getLogger(__name__)

class DataDistribution(Enum):
    """Data distribution types"""
    NORMAL = "normal"
    UNIFORM = "uniform"
    EXPONENTIAL = "exponential"
    CATEGORICAL = "categorical"
    BOOLEAN = "boolean"
    DATETIME = "datetime"

@dataclass
class FieldSimulationConfig:
    """Configuration for simulating a specific field"""
    field_name: str
    field_type: str  # string, integer, float, boolean, datetime
    distribution: DataDistribution
    parameters: Dict[str, Any]  # Distribution-specific parameters
    constraints: Dict[str, Any] = field(default_factory=dict)  # Min, max, allowed_values, etc.
    correlation_fields: List[str] = field(default_factory=list)  # Fields this correlates with
    correlation_strength: float = 0.0  # -1 to 1
    null_probability: float = 0.0  # Probability of null values

@dataclass
class TemplateSimulationConfig:
    """Configuration for simulating data for a template"""
    template_id: str
    field_configs: List[FieldSimulationConfig]
    scenarios: Dict[str, Dict[str, Any]] = field(default_factory=dict)  # Named scenarios
    data_quality_issues: Dict[str, float] = field(default_factory=dict)  # Issue probabilities
    temporal_patterns: Dict[str, Any] = field(default_factory=dict)  # Time-based patterns
    business_rules: List[Dict[str, Any]] = field(default_factory=list)  # Business logic rules

@dataclass
class SimulationResult:
    """Result of data simulation"""
    simulation_id: str
    template_id: str
    scenario: str
    record_count: int
    data: List[Dict[str, Any]]
    metadata: Dict[str, Any]
    quality_metrics: Dict[str, Any]
    created_at: datetime = field(default_factory=datetime.utcnow)

class TemplateDataSimulator:
    """
    Comprehensive data simulator for RBIA templates
    
    Provides:
    - Realistic data generation for each template
    - Multiple business scenarios
    - Data quality issues simulation
    - Temporal patterns and seasonality
    - Correlation between fields
    - Industry-specific data patterns
    """
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.template_registry = IndustryTemplateRegistry()
        self.simulation_configs: Dict[str, TemplateSimulationConfig] = {}
        self._initialize_simulation_configs()
        
        # Seed for reproducible results
        random.seed(42)
        np.random.seed(42)
    
    def _initialize_simulation_configs(self):
        """Initialize simulation configurations for all templates"""
        
        # SaaS templates
        self._create_saas_simulation_configs()
        
        # Banking templates
        self._create_banking_simulation_configs()
        
        # Insurance templates
        self._create_insurance_simulation_configs()
        
        # E-commerce templates
        self._create_ecommerce_simulation_configs()
        
        # Financial Services templates
        self._create_financial_services_simulation_configs()
        
        self.logger.info(f"Initialized simulation configs for {len(self.simulation_configs)} templates")
    
    def _create_saas_simulation_configs(self):
        """Create simulation configurations for SaaS templates"""
        
        # SaaS Churn Risk Alert
        self.simulation_configs["saas_churn_risk_alert"] = TemplateSimulationConfig(
            template_id="saas_churn_risk_alert",
            field_configs=[
                FieldSimulationConfig(
                    field_name="customer_id",
                    field_type="string",
                    distribution=DataDistribution.CATEGORICAL,
                    parameters={"prefix": "CUST-", "number_range": [1000, 9999]}
                ),
                FieldSimulationConfig(
                    field_name="customer_tenure_months",
                    field_type="integer",
                    distribution=DataDistribution.EXPONENTIAL,
                    parameters={"lambda": 0.1, "min": 1, "max": 120},
                    constraints={"min": 1, "max": 120}
                ),
                FieldSimulationConfig(
                    field_name="mrr",
                    field_type="float",
                    distribution=DataDistribution.NORMAL,
                    parameters={"mean": 500, "std": 300},
                    constraints={"min": 50, "max": 5000},
                    correlation_fields=["customer_tenure_months"],
                    correlation_strength=0.3
                ),
                FieldSimulationConfig(
                    field_name="usage_frequency",
                    field_type="float",
                    distribution=DataDistribution.NORMAL,
                    parameters={"mean": 0.6, "std": 0.25},
                    constraints={"min": 0.0, "max": 1.0},
                    correlation_fields=["mrr"],
                    correlation_strength=0.4
                ),
                FieldSimulationConfig(
                    field_name="support_tickets_last_30d",
                    field_type="integer",
                    distribution=DataDistribution.EXPONENTIAL,
                    parameters={"lambda": 0.5, "min": 0, "max": 20},
                    constraints={"min": 0, "max": 20},
                    correlation_fields=["usage_frequency"],
                    correlation_strength=-0.3
                ),
                FieldSimulationConfig(
                    field_name="last_login_days_ago",
                    field_type="integer",
                    distribution=DataDistribution.EXPONENTIAL,
                    parameters={"lambda": 0.1, "min": 0, "max": 90},
                    constraints={"min": 0, "max": 90},
                    correlation_fields=["usage_frequency"],
                    correlation_strength=-0.6
                ),
                FieldSimulationConfig(
                    field_name="feature_adoption_score",
                    field_type="float",
                    distribution=DataDistribution.NORMAL,
                    parameters={"mean": 0.5, "std": 0.2},
                    constraints={"min": 0.0, "max": 1.0},
                    correlation_fields=["usage_frequency", "customer_tenure_months"],
                    correlation_strength=0.5
                ),
                FieldSimulationConfig(
                    field_name="payment_method",
                    field_type="string",
                    distribution=DataDistribution.CATEGORICAL,
                    parameters={"values": ["credit_card", "bank_transfer", "paypal", "invoice"], 
                               "weights": [0.6, 0.2, 0.15, 0.05]}
                ),
                FieldSimulationConfig(
                    field_name="contract_type",
                    field_type="string",
                    distribution=DataDistribution.CATEGORICAL,
                    parameters={"values": ["monthly", "annual", "multi_year"], 
                               "weights": [0.5, 0.4, 0.1]}
                )
            ],
            scenarios={
                "high_churn_risk": {
                    "description": "Customers with high churn probability",
                    "field_adjustments": {
                        "mrr": {"mean": 200, "std": 100},
                        "usage_frequency": {"mean": 0.2, "std": 0.1},
                        "last_login_days_ago": {"mean": 45, "std": 20},
                        "support_tickets_last_30d": {"lambda": 0.3}
                    }
                },
                "healthy_customers": {
                    "description": "Engaged, low-churn customers",
                    "field_adjustments": {
                        "mrr": {"mean": 800, "std": 400},
                        "usage_frequency": {"mean": 0.8, "std": 0.15},
                        "last_login_days_ago": {"mean": 2, "std": 3},
                        "feature_adoption_score": {"mean": 0.7, "std": 0.2}
                    }
                },
                "new_customers": {
                    "description": "Recently acquired customers",
                    "field_adjustments": {
                        "customer_tenure_months": {"lambda": 0.5, "max": 12},
                        "feature_adoption_score": {"mean": 0.3, "std": 0.15}
                    }
                }
            },
            data_quality_issues={
                "missing_values": 0.02,
                "outliers": 0.01,
                "inconsistent_formats": 0.005
            },
            temporal_patterns={
                "seasonal_churn": {
                    "pattern": "quarterly",
                    "peak_months": [3, 6, 9, 12],
                    "amplitude": 0.3
                }
            }
        )
        
        # SaaS Forecast Variance Detector
        self.simulation_configs["saas_forecast_variance_detector"] = TemplateSimulationConfig(
            template_id="saas_forecast_variance_detector",
            field_configs=[
                FieldSimulationConfig(
                    field_name="period_id",
                    field_type="string",
                    distribution=DataDistribution.CATEGORICAL,
                    parameters={"format": "YYYY-MM", "periods": 24}
                ),
                FieldSimulationConfig(
                    field_name="forecasted_mrr",
                    field_type="float",
                    distribution=DataDistribution.NORMAL,
                    parameters={"mean": 100000, "std": 20000},
                    constraints={"min": 50000, "max": 500000}
                ),
                FieldSimulationConfig(
                    field_name="actual_mrr",
                    field_type="float",
                    distribution=DataDistribution.NORMAL,
                    parameters={"mean": 100000, "std": 25000},
                    constraints={"min": 50000, "max": 500000},
                    correlation_fields=["forecasted_mrr"],
                    correlation_strength=0.8
                ),
                FieldSimulationConfig(
                    field_name="forecasted_new_customers",
                    field_type="integer",
                    distribution=DataDistribution.NORMAL,
                    parameters={"mean": 50, "std": 15},
                    constraints={"min": 10, "max": 200}
                ),
                FieldSimulationConfig(
                    field_name="actual_new_customers",
                    field_type="integer",
                    distribution=DataDistribution.NORMAL,
                    parameters={"mean": 50, "std": 18},
                    constraints={"min": 5, "max": 200},
                    correlation_fields=["forecasted_new_customers"],
                    correlation_strength=0.7
                ),
                FieldSimulationConfig(
                    field_name="forecasted_churn_rate",
                    field_type="float",
                    distribution=DataDistribution.NORMAL,
                    parameters={"mean": 0.05, "std": 0.01},
                    constraints={"min": 0.01, "max": 0.15}
                ),
                FieldSimulationConfig(
                    field_name="actual_churn_rate",
                    field_type="float",
                    distribution=DataDistribution.NORMAL,
                    parameters={"mean": 0.05, "std": 0.015},
                    constraints={"min": 0.01, "max": 0.15},
                    correlation_fields=["forecasted_churn_rate"],
                    correlation_strength=0.6
                ),
                FieldSimulationConfig(
                    field_name="seasonal_factor",
                    field_type="float",
                    distribution=DataDistribution.NORMAL,
                    parameters={"mean": 1.0, "std": 0.1},
                    constraints={"min": 0.8, "max": 1.3}
                ),
                FieldSimulationConfig(
                    field_name="market_conditions",
                    field_type="string",
                    distribution=DataDistribution.CATEGORICAL,
                    parameters={"values": ["favorable", "neutral", "challenging"], 
                               "weights": [0.3, 0.5, 0.2]}
                )
            ],
            scenarios={
                "accurate_forecasts": {
                    "description": "Periods with accurate forecasting",
                    "field_adjustments": {
                        "actual_mrr": {"correlation_strength": 0.95},
                        "actual_new_customers": {"correlation_strength": 0.9},
                        "actual_churn_rate": {"correlation_strength": 0.85}
                    }
                },
                "forecast_variance": {
                    "description": "Periods with significant forecast variance",
                    "field_adjustments": {
                        "actual_mrr": {"std": 40000, "correlation_strength": 0.5},
                        "actual_new_customers": {"std": 25, "correlation_strength": 0.4},
                        "actual_churn_rate": {"std": 0.025, "correlation_strength": 0.3}
                    }
                }
            }
        )
    
    def _create_banking_simulation_configs(self):
        """Create simulation configurations for Banking templates"""
        
        # Banking Credit Scoring Check
        self.simulation_configs["banking_credit_scoring_check"] = TemplateSimulationConfig(
            template_id="banking_credit_scoring_check",
            field_configs=[
                FieldSimulationConfig(
                    field_name="application_id",
                    field_type="string",
                    distribution=DataDistribution.CATEGORICAL,
                    parameters={"prefix": "APP-", "number_range": [100000, 999999]}
                ),
                FieldSimulationConfig(
                    field_name="applicant_id",
                    field_type="string",
                    distribution=DataDistribution.CATEGORICAL,
                    parameters={"prefix": "APPL-", "number_range": [10000, 99999]}
                ),
                FieldSimulationConfig(
                    field_name="credit_history_length",
                    field_type="integer",
                    distribution=DataDistribution.NORMAL,
                    parameters={"mean": 60, "std": 30},
                    constraints={"min": 0, "max": 240}
                ),
                FieldSimulationConfig(
                    field_name="annual_income",
                    field_type="float",
                    distribution=DataDistribution.NORMAL,
                    parameters={"mean": 65000, "std": 25000},
                    constraints={"min": 20000, "max": 200000}
                ),
                FieldSimulationConfig(
                    field_name="debt_to_income_ratio",
                    field_type="float",
                    distribution=DataDistribution.NORMAL,
                    parameters={"mean": 0.3, "std": 0.15},
                    constraints={"min": 0.0, "max": 0.8},
                    correlation_fields=["annual_income"],
                    correlation_strength=-0.2
                ),
                FieldSimulationConfig(
                    field_name="employment_status",
                    field_type="string",
                    distribution=DataDistribution.CATEGORICAL,
                    parameters={"values": ["employed", "self_employed", "unemployed", "retired"], 
                               "weights": [0.7, 0.15, 0.1, 0.05]}
                ),
                FieldSimulationConfig(
                    field_name="existing_credit_accounts",
                    field_type="integer",
                    distribution=DataDistribution.NORMAL,
                    parameters={"mean": 3, "std": 2},
                    constraints={"min": 0, "max": 15},
                    correlation_fields=["credit_history_length"],
                    correlation_strength=0.4
                ),
                FieldSimulationConfig(
                    field_name="payment_history_score",
                    field_type="float",
                    distribution=DataDistribution.NORMAL,
                    parameters={"mean": 0.8, "std": 0.2},
                    constraints={"min": 0.0, "max": 1.0},
                    correlation_fields=["credit_history_length"],
                    correlation_strength=0.3
                ),
                FieldSimulationConfig(
                    field_name="credit_utilization_ratio",
                    field_type="float",
                    distribution=DataDistribution.NORMAL,
                    parameters={"mean": 0.3, "std": 0.2},
                    constraints={"min": 0.0, "max": 1.0},
                    correlation_fields=["payment_history_score"],
                    correlation_strength=-0.4
                ),
                FieldSimulationConfig(
                    field_name="loan_amount",
                    field_type="float",
                    distribution=DataDistribution.NORMAL,
                    parameters={"mean": 25000, "std": 15000},
                    constraints={"min": 1000, "max": 100000},
                    correlation_fields=["annual_income"],
                    correlation_strength=0.5
                ),
                FieldSimulationConfig(
                    field_name="loan_purpose",
                    field_type="string",
                    distribution=DataDistribution.CATEGORICAL,
                    parameters={"values": ["home", "auto", "personal", "business", "education"], 
                               "weights": [0.4, 0.25, 0.2, 0.1, 0.05]}
                )
            ],
            scenarios={
                "high_creditworthy": {
                    "description": "Highly creditworthy applicants",
                    "field_adjustments": {
                        "annual_income": {"mean": 85000, "std": 20000},
                        "credit_history_length": {"mean": 80, "std": 25},
                        "payment_history_score": {"mean": 0.9, "std": 0.1},
                        "debt_to_income_ratio": {"mean": 0.2, "std": 0.1},
                        "credit_utilization_ratio": {"mean": 0.2, "std": 0.1}
                    }
                },
                "risky_applicants": {
                    "description": "High-risk loan applicants",
                    "field_adjustments": {
                        "annual_income": {"mean": 40000, "std": 15000},
                        "credit_history_length": {"mean": 20, "std": 15},
                        "payment_history_score": {"mean": 0.5, "std": 0.2},
                        "debt_to_income_ratio": {"mean": 0.5, "std": 0.15},
                        "credit_utilization_ratio": {"mean": 0.6, "std": 0.2}
                    }
                }
            }
        )
        
        # Banking Fraudulent Disbursal Detector
        self.simulation_configs["banking_fraudulent_disbursal_detector"] = TemplateSimulationConfig(
            template_id="banking_fraudulent_disbursal_detector",
            field_configs=[
                FieldSimulationConfig(
                    field_name="transaction_id",
                    field_type="string",
                    distribution=DataDistribution.CATEGORICAL,
                    parameters={"prefix": "TXN-", "number_range": [1000000, 9999999]}
                ),
                FieldSimulationConfig(
                    field_name="transaction_amount",
                    field_type="float",
                    distribution=DataDistribution.EXPONENTIAL,
                    parameters={"lambda": 0.0001, "min": 100, "max": 100000},
                    constraints={"min": 100, "max": 100000}
                ),
                FieldSimulationConfig(
                    field_name="recipient_account_age",
                    field_type="integer",
                    distribution=DataDistribution.EXPONENTIAL,
                    parameters={"lambda": 0.01, "min": 1, "max": 3650},
                    constraints={"min": 1, "max": 3650}
                ),
                FieldSimulationConfig(
                    field_name="sender_account_history",
                    field_type="float",
                    distribution=DataDistribution.NORMAL,
                    parameters={"mean": 0.7, "std": 0.2},
                    constraints={"min": 0.0, "max": 1.0}
                ),
                FieldSimulationConfig(
                    field_name="transaction_time",
                    field_type="datetime",
                    distribution=DataDistribution.DATETIME,
                    parameters={"start": "2023-01-01", "end": "2023-12-31", "pattern": "business_hours"}
                ),
                FieldSimulationConfig(
                    field_name="geographic_location",
                    field_type="string",
                    distribution=DataDistribution.CATEGORICAL,
                    parameters={"values": ["domestic", "international_low_risk", "international_high_risk"], 
                               "weights": [0.8, 0.15, 0.05]}
                ),
                FieldSimulationConfig(
                    field_name="device_fingerprint",
                    field_type="string",
                    distribution=DataDistribution.CATEGORICAL,
                    parameters={"prefix": "DEV-", "number_range": [100000, 999999]}
                ),
                FieldSimulationConfig(
                    field_name="transaction_pattern_score",
                    field_type="float",
                    distribution=DataDistribution.NORMAL,
                    parameters={"mean": 0.6, "std": 0.2},
                    constraints={"min": 0.0, "max": 1.0}
                ),
                FieldSimulationConfig(
                    field_name="recipient_risk_score",
                    field_type="float",
                    distribution=DataDistribution.NORMAL,
                    parameters={"mean": 0.3, "std": 0.2},
                    constraints={"min": 0.0, "max": 1.0},
                    correlation_fields=["recipient_account_age"],
                    correlation_strength=-0.4
                ),
                FieldSimulationConfig(
                    field_name="velocity_check_score",
                    field_type="float",
                    distribution=DataDistribution.NORMAL,
                    parameters={"mean": 0.5, "std": 0.3},
                    constraints={"min": 0.0, "max": 1.0}
                )
            ],
            scenarios={
                "legitimate_transactions": {
                    "description": "Normal, legitimate transactions",
                    "field_adjustments": {
                        "transaction_amount": {"lambda": 0.0002, "max": 50000},
                        "recipient_risk_score": {"mean": 0.2, "std": 0.1},
                        "transaction_pattern_score": {"mean": 0.7, "std": 0.15},
                        "geographic_location": {"weights": [0.9, 0.08, 0.02]}
                    }
                },
                "suspicious_transactions": {
                    "description": "Potentially fraudulent transactions",
                    "field_adjustments": {
                        "transaction_amount": {"lambda": 0.00005, "min": 5000},
                        "recipient_account_age": {"lambda": 0.05, "max": 90},
                        "recipient_risk_score": {"mean": 0.7, "std": 0.2},
                        "transaction_pattern_score": {"mean": 0.3, "std": 0.2},
                        "geographic_location": {"weights": [0.3, 0.3, 0.4]}
                    }
                }
            }
        )
    
    def _create_insurance_simulation_configs(self):
        """Create simulation configurations for Insurance templates"""
        
        # Insurance Claim Fraud Anomaly
        self.simulation_configs["insurance_claim_fraud_anomaly"] = TemplateSimulationConfig(
            template_id="insurance_claim_fraud_anomaly",
            field_configs=[
                FieldSimulationConfig(
                    field_name="claim_id",
                    field_type="string",
                    distribution=DataDistribution.CATEGORICAL,
                    parameters={"prefix": "CLM-", "number_range": [100000, 999999]}
                ),
                FieldSimulationConfig(
                    field_name="claim_amount",
                    field_type="float",
                    distribution=DataDistribution.EXPONENTIAL,
                    parameters={"lambda": 0.0001, "min": 500, "max": 50000},
                    constraints={"min": 500, "max": 50000}
                ),
                FieldSimulationConfig(
                    field_name="policy_age",
                    field_type="integer",
                    distribution=DataDistribution.EXPONENTIAL,
                    parameters={"lambda": 0.01, "min": 30, "max": 3650},
                    constraints={"min": 30, "max": 3650}
                ),
                FieldSimulationConfig(
                    field_name="claimant_history",
                    field_type="float",
                    distribution=DataDistribution.NORMAL,
                    parameters={"mean": 0.7, "std": 0.2},
                    constraints={"min": 0.0, "max": 1.0}
                ),
                FieldSimulationConfig(
                    field_name="incident_location",
                    field_type="string",
                    distribution=DataDistribution.CATEGORICAL,
                    parameters={"values": ["home", "work", "public", "vehicle", "other"], 
                               "weights": [0.4, 0.2, 0.2, 0.15, 0.05]}
                ),
                FieldSimulationConfig(
                    field_name="incident_time",
                    field_type="datetime",
                    distribution=DataDistribution.DATETIME,
                    parameters={"start": "2023-01-01", "end": "2023-12-31", "pattern": "random"}
                ),
                FieldSimulationConfig(
                    field_name="claim_description_sentiment",
                    field_type="float",
                    distribution=DataDistribution.NORMAL,
                    parameters={"mean": 0.0, "std": 0.3},
                    constraints={"min": -1.0, "max": 1.0}
                ),
                FieldSimulationConfig(
                    field_name="supporting_documents_count",
                    field_type="integer",
                    distribution=DataDistribution.NORMAL,
                    parameters={"mean": 3, "std": 2},
                    constraints={"min": 0, "max": 10},
                    correlation_fields=["claim_amount"],
                    correlation_strength=0.3
                ),
                FieldSimulationConfig(
                    field_name="witness_statements_count",
                    field_type="integer",
                    distribution=DataDistribution.EXPONENTIAL,
                    parameters={"lambda": 1.0, "min": 0, "max": 5},
                    constraints={"min": 0, "max": 5}
                ),
                FieldSimulationConfig(
                    field_name="medical_provider_history",
                    field_type="float",
                    distribution=DataDistribution.NORMAL,
                    parameters={"mean": 0.8, "std": 0.2},
                    constraints={"min": 0.0, "max": 1.0}
                ),
                FieldSimulationConfig(
                    field_name="repair_shop_history",
                    field_type="float",
                    distribution=DataDistribution.NORMAL,
                    parameters={"mean": 0.7, "std": 0.25},
                    constraints={"min": 0.0, "max": 1.0}
                )
            ],
            scenarios={
                "legitimate_claims": {
                    "description": "Legitimate insurance claims",
                    "field_adjustments": {
                        "claim_amount": {"lambda": 0.0002, "max": 25000},
                        "supporting_documents_count": {"mean": 4, "std": 1.5},
                        "witness_statements_count": {"lambda": 0.7},
                        "medical_provider_history": {"mean": 0.85, "std": 0.15},
                        "claimant_history": {"mean": 0.8, "std": 0.15}
                    }
                },
                "suspicious_claims": {
                    "description": "Potentially fraudulent claims",
                    "field_adjustments": {
                        "claim_amount": {"lambda": 0.00005, "min": 10000},
                        "policy_age": {"lambda": 0.05, "max": 180},
                        "supporting_documents_count": {"mean": 1.5, "std": 1},
                        "witness_statements_count": {"lambda": 2.0},
                        "medical_provider_history": {"mean": 0.4, "std": 0.3},
                        "claimant_history": {"mean": 0.4, "std": 0.2}
                    }
                }
            }
        )
    
    def _create_ecommerce_simulation_configs(self):
        """Create simulation configurations for E-commerce templates"""
        
        # E-commerce Checkout Fraud Scoring
        self.simulation_configs["ecommerce_checkout_fraud_scoring"] = TemplateSimulationConfig(
            template_id="ecommerce_checkout_fraud_scoring",
            field_configs=[
                FieldSimulationConfig(
                    field_name="order_id",
                    field_type="string",
                    distribution=DataDistribution.CATEGORICAL,
                    parameters={"prefix": "ORD-", "number_range": [1000000, 9999999]}
                ),
                FieldSimulationConfig(
                    field_name="transaction_amount",
                    field_type="float",
                    distribution=DataDistribution.EXPONENTIAL,
                    parameters={"lambda": 0.01, "min": 10, "max": 2000},
                    constraints={"min": 10, "max": 2000}
                ),
                FieldSimulationConfig(
                    field_name="customer_account_age",
                    field_type="integer",
                    distribution=DataDistribution.EXPONENTIAL,
                    parameters={"lambda": 0.005, "min": 1, "max": 2000},
                    constraints={"min": 1, "max": 2000}
                ),
                FieldSimulationConfig(
                    field_name="shipping_billing_match",
                    field_type="boolean",
                    distribution=DataDistribution.BOOLEAN,
                    parameters={"probability": 0.85}
                ),
                FieldSimulationConfig(
                    field_name="payment_method",
                    field_type="string",
                    distribution=DataDistribution.CATEGORICAL,
                    parameters={"values": ["credit_card", "debit_card", "paypal", "apple_pay", "crypto"], 
                               "weights": [0.5, 0.25, 0.15, 0.08, 0.02]}
                ),
                FieldSimulationConfig(
                    field_name="device_fingerprint",
                    field_type="string",
                    distribution=DataDistribution.CATEGORICAL,
                    parameters={"prefix": "FP-", "number_range": [100000, 999999]}
                ),
                FieldSimulationConfig(
                    field_name="ip_geolocation",
                    field_type="string",
                    distribution=DataDistribution.CATEGORICAL,
                    parameters={"values": ["domestic", "international_safe", "international_risky", "proxy/vpn"], 
                               "weights": [0.7, 0.2, 0.08, 0.02]}
                ),
                FieldSimulationConfig(
                    field_name="order_velocity",
                    field_type="float",
                    distribution=DataDistribution.EXPONENTIAL,
                    parameters={"lambda": 2.0, "min": 0.1, "max": 10.0},
                    constraints={"min": 0.1, "max": 10.0}
                ),
                FieldSimulationConfig(
                    field_name="product_category_risk",
                    field_type="float",
                    distribution=DataDistribution.NORMAL,
                    parameters={"mean": 0.3, "std": 0.2},
                    constraints={"min": 0.0, "max": 1.0}
                ),
                FieldSimulationConfig(
                    field_name="customer_purchase_history",
                    field_type="float",
                    distribution=DataDistribution.NORMAL,
                    parameters={"mean": 0.6, "std": 0.3},
                    constraints={"min": 0.0, "max": 1.0},
                    correlation_fields=["customer_account_age"],
                    correlation_strength=0.5
                ),
                FieldSimulationConfig(
                    field_name="time_of_purchase",
                    field_type="datetime",
                    distribution=DataDistribution.DATETIME,
                    parameters={"start": "2023-01-01", "end": "2023-12-31", "pattern": "shopping_hours"}
                )
            ],
            scenarios={
                "normal_shopping": {
                    "description": "Normal customer shopping behavior",
                    "field_adjustments": {
                        "transaction_amount": {"lambda": 0.02, "max": 500},
                        "shipping_billing_match": {"probability": 0.9},
                        "order_velocity": {"lambda": 3.0, "max": 2.0},
                        "ip_geolocation": {"weights": [0.8, 0.18, 0.015, 0.005]}
                    }
                },
                "fraud_attempts": {
                    "description": "Fraudulent purchase attempts",
                    "field_adjustments": {
                        "transaction_amount": {"lambda": 0.002, "min": 200},
                        "customer_account_age": {"lambda": 0.1, "max": 30},
                        "shipping_billing_match": {"probability": 0.3},
                        "order_velocity": {"lambda": 0.5, "min": 2.0},
                        "ip_geolocation": {"weights": [0.2, 0.2, 0.3, 0.3]},
                        "customer_purchase_history": {"mean": 0.1, "std": 0.1}
                    }
                }
            }
        )
    
    def _create_financial_services_simulation_configs(self):
        """Create simulation configurations for Financial Services templates"""
        
        # FS Liquidity Risk Early Warning
        self.simulation_configs["fs_liquidity_risk_early_warning"] = TemplateSimulationConfig(
            template_id="fs_liquidity_risk_early_warning",
            field_configs=[
                FieldSimulationConfig(
                    field_name="institution_id",
                    field_type="string",
                    distribution=DataDistribution.CATEGORICAL,
                    parameters={"prefix": "INST-", "number_range": [1000, 9999]}
                ),
                FieldSimulationConfig(
                    field_name="cash_reserves_ratio",
                    field_type="float",
                    distribution=DataDistribution.NORMAL,
                    parameters={"mean": 0.12, "std": 0.03},
                    constraints={"min": 0.05, "max": 0.25}
                ),
                FieldSimulationConfig(
                    field_name="deposit_withdrawal_velocity",
                    field_type="float",
                    distribution=DataDistribution.NORMAL,
                    parameters={"mean": 0.02, "std": 0.01},
                    constraints={"min": 0.001, "max": 0.1}
                ),
                FieldSimulationConfig(
                    field_name="loan_to_deposit_ratio",
                    field_type="float",
                    distribution=DataDistribution.NORMAL,
                    parameters={"mean": 0.8, "std": 0.1},
                    constraints={"min": 0.5, "max": 1.0}
                ),
                FieldSimulationConfig(
                    field_name="interbank_borrowing_rate",
                    field_type="float",
                    distribution=DataDistribution.NORMAL,
                    parameters={"mean": 0.03, "std": 0.01},
                    constraints={"min": 0.01, "max": 0.08}
                ),
                FieldSimulationConfig(
                    field_name="market_volatility_index",
                    field_type="float",
                    distribution=DataDistribution.NORMAL,
                    parameters={"mean": 20, "std": 8},
                    constraints={"min": 5, "max": 50}
                ),
                FieldSimulationConfig(
                    field_name="regulatory_capital_ratio",
                    field_type="float",
                    distribution=DataDistribution.NORMAL,
                    parameters={"mean": 0.15, "std": 0.03},
                    constraints={"min": 0.08, "max": 0.25}
                ),
                FieldSimulationConfig(
                    field_name="stress_test_results",
                    field_type="float",
                    distribution=DataDistribution.NORMAL,
                    parameters={"mean": 0.7, "std": 0.15},
                    constraints={"min": 0.3, "max": 1.0}
                ),
                FieldSimulationConfig(
                    field_name="funding_concentration",
                    field_type="float",
                    distribution=DataDistribution.NORMAL,
                    parameters={"mean": 0.4, "std": 0.15},
                    constraints={"min": 0.1, "max": 0.8}
                ),
                FieldSimulationConfig(
                    field_name="maturity_mismatch_ratio",
                    field_type="float",
                    distribution=DataDistribution.NORMAL,
                    parameters={"mean": 0.3, "std": 0.1},
                    constraints={"min": 0.1, "max": 0.6}
                ),
                FieldSimulationConfig(
                    field_name="liquidity_coverage_ratio",
                    field_type="float",
                    distribution=DataDistribution.NORMAL,
                    parameters={"mean": 1.2, "std": 0.2},
                    constraints={"min": 0.8, "max": 2.0}
                )
            ],
            scenarios={
                "stable_liquidity": {
                    "description": "Institutions with stable liquidity",
                    "field_adjustments": {
                        "cash_reserves_ratio": {"mean": 0.15, "std": 0.02},
                        "deposit_withdrawal_velocity": {"mean": 0.015, "std": 0.005},
                        "liquidity_coverage_ratio": {"mean": 1.4, "std": 0.15},
                        "stress_test_results": {"mean": 0.8, "std": 0.1}
                    }
                },
                "liquidity_stress": {
                    "description": "Institutions under liquidity stress",
                    "field_adjustments": {
                        "cash_reserves_ratio": {"mean": 0.08, "std": 0.02},
                        "deposit_withdrawal_velocity": {"mean": 0.04, "std": 0.015},
                        "liquidity_coverage_ratio": {"mean": 0.95, "std": 0.1},
                        "market_volatility_index": {"mean": 35, "std": 10},
                        "stress_test_results": {"mean": 0.5, "std": 0.15}
                    }
                }
            }
        )
    
    async def simulate_data(
        self,
        template_id: str,
        record_count: int = 1000,
        scenario: str = "default",
        include_quality_issues: bool = True
    ) -> SimulationResult:
        """Simulate data for a template"""
        
        try:
            config = self.simulation_configs.get(template_id)
            if not config:
                raise ValueError(f"No simulation config found for template {template_id}")
            
            simulation_id = str(uuid.uuid4())
            
            # Apply scenario adjustments if specified
            field_configs = self._apply_scenario_adjustments(config, scenario)
            
            # Generate data
            data = []
            for i in range(record_count):
                record = await self._generate_record(field_configs, i)
                
                # Apply data quality issues if enabled
                if include_quality_issues:
                    record = self._apply_quality_issues(record, config.data_quality_issues)
                
                data.append(record)
            
            # Calculate quality metrics
            quality_metrics = self._calculate_quality_metrics(data, config)
            
            # Create metadata
            metadata = {
                "template_id": template_id,
                "scenario": scenario,
                "record_count": record_count,
                "generation_method": "statistical_simulation",
                "correlation_applied": any(fc.correlation_fields for fc in field_configs),
                "quality_issues_included": include_quality_issues,
                "field_count": len(field_configs)
            }
            
            result = SimulationResult(
                simulation_id=simulation_id,
                template_id=template_id,
                scenario=scenario,
                record_count=record_count,
                data=data,
                metadata=metadata,
                quality_metrics=quality_metrics
            )
            
            self.logger.info(f"Generated {record_count} records for template {template_id} (scenario: {scenario})")
            
            return result
            
        except Exception as e:
            self.logger.error(f"Failed to simulate data for template {template_id}: {e}")
            raise
    
    def _apply_scenario_adjustments(
        self,
        config: TemplateSimulationConfig,
        scenario: str
    ) -> List[FieldSimulationConfig]:
        """Apply scenario-specific adjustments to field configurations"""
        
        field_configs = config.field_configs.copy()
        
        if scenario != "default" and scenario in config.scenarios:
            scenario_config = config.scenarios[scenario]
            adjustments = scenario_config.get("field_adjustments", {})
            
            for field_config in field_configs:
                if field_config.field_name in adjustments:
                    adjustment = adjustments[field_config.field_name]
                    
                    # Apply parameter adjustments
                    for param_name, param_value in adjustment.items():
                        if param_name in field_config.parameters:
                            field_config.parameters[param_name] = param_value
        
        return field_configs
    
    async def _generate_record(
        self,
        field_configs: List[FieldSimulationConfig],
        record_index: int
    ) -> Dict[str, Any]:
        """Generate a single data record"""
        
        record = {}
        correlation_context = {}
        
        # First pass: generate independent fields
        for field_config in field_configs:
            if not field_config.correlation_fields:
                value = self._generate_field_value(field_config, record_index, correlation_context)
                record[field_config.field_name] = value
                correlation_context[field_config.field_name] = value
        
        # Second pass: generate correlated fields
        for field_config in field_configs:
            if field_config.correlation_fields:
                value = self._generate_field_value(field_config, record_index, correlation_context)
                record[field_config.field_name] = value
                correlation_context[field_config.field_name] = value
        
        return record
    
    def _generate_field_value(
        self,
        field_config: FieldSimulationConfig,
        record_index: int,
        correlation_context: Dict[str, Any]
    ) -> Any:
        """Generate value for a specific field"""
        
        # Check for null values
        if random.random() < field_config.null_probability:
            return None
        
        # Apply correlation if specified
        base_value = self._generate_base_value(field_config, record_index)
        
        if field_config.correlation_fields and field_config.correlation_strength != 0:
            base_value = self._apply_correlation(
                base_value, field_config, correlation_context
            )
        
        # Apply constraints
        return self._apply_constraints(base_value, field_config)
    
    def _generate_base_value(self, field_config: FieldSimulationConfig, record_index: int) -> Any:
        """Generate base value based on distribution"""
        
        if field_config.distribution == DataDistribution.NORMAL:
            mean = field_config.parameters.get("mean", 0)
            std = field_config.parameters.get("std", 1)
            return np.random.normal(mean, std)
        
        elif field_config.distribution == DataDistribution.UNIFORM:
            min_val = field_config.parameters.get("min", 0)
            max_val = field_config.parameters.get("max", 1)
            return np.random.uniform(min_val, max_val)
        
        elif field_config.distribution == DataDistribution.EXPONENTIAL:
            lambda_param = field_config.parameters.get("lambda", 1.0)
            min_val = field_config.parameters.get("min", 0)
            max_val = field_config.parameters.get("max", float('inf'))
            value = np.random.exponential(1/lambda_param) + min_val
            return min(value, max_val)
        
        elif field_config.distribution == DataDistribution.CATEGORICAL:
            values = field_config.parameters.get("values", [])
            weights = field_config.parameters.get("weights")
            
            if "prefix" in field_config.parameters:
                # Generate ID-like values
                prefix = field_config.parameters["prefix"]
                number_range = field_config.parameters.get("number_range", [1000, 9999])
                number = random.randint(number_range[0], number_range[1])
                return f"{prefix}{number}"
            
            elif "format" in field_config.parameters:
                # Generate formatted values (e.g., dates)
                format_type = field_config.parameters["format"]
                if format_type == "YYYY-MM":
                    periods = field_config.parameters.get("periods", 12)
                    base_date = datetime(2023, 1, 1)
                    month_offset = record_index % periods
                    date = base_date + timedelta(days=30 * month_offset)
                    return date.strftime("%Y-%m")
            
            else:
                return np.random.choice(values, p=weights)
        
        elif field_config.distribution == DataDistribution.BOOLEAN:
            probability = field_config.parameters.get("probability", 0.5)
            return random.random() < probability
        
        elif field_config.distribution == DataDistribution.DATETIME:
            start_str = field_config.parameters.get("start", "2023-01-01")
            end_str = field_config.parameters.get("end", "2023-12-31")
            pattern = field_config.parameters.get("pattern", "random")
            
            start_date = datetime.strptime(start_str, "%Y-%m-%d")
            end_date = datetime.strptime(end_str, "%Y-%m-%d")
            
            if pattern == "business_hours":
                # Generate during business hours
                random_date = start_date + timedelta(
                    seconds=random.randint(0, int((end_date - start_date).total_seconds()))
                )
                # Adjust to business hours (9 AM - 5 PM)
                random_date = random_date.replace(
                    hour=random.randint(9, 17),
                    minute=random.randint(0, 59),
                    second=random.randint(0, 59)
                )
                return random_date
            
            elif pattern == "shopping_hours":
                # Generate during typical shopping hours
                random_date = start_date + timedelta(
                    seconds=random.randint(0, int((end_date - start_date).total_seconds()))
                )
                # Adjust to shopping hours (8 AM - 10 PM)
                random_date = random_date.replace(
                    hour=random.randint(8, 22),
                    minute=random.randint(0, 59),
                    second=random.randint(0, 59)
                )
                return random_date
            
            else:
                # Random datetime
                return start_date + timedelta(
                    seconds=random.randint(0, int((end_date - start_date).total_seconds()))
                )
        
        return 0  # Default fallback
    
    def _apply_correlation(
        self,
        base_value: Any,
        field_config: FieldSimulationConfig,
        correlation_context: Dict[str, Any]
    ) -> Any:
        """Apply correlation with other fields"""
        
        if not isinstance(base_value, (int, float)):
            return base_value
        
        correlation_adjustment = 0
        
        for corr_field in field_config.correlation_fields:
            if corr_field in correlation_context:
                corr_value = correlation_context[corr_field]
                
                if isinstance(corr_value, (int, float)):
                    # Normalize correlation value
                    corr_normalized = (corr_value - field_config.parameters.get("mean", 0)) / field_config.parameters.get("std", 1)
                    correlation_adjustment += corr_normalized * field_config.correlation_strength
        
        # Apply correlation adjustment
        if field_config.distribution == DataDistribution.NORMAL:
            std = field_config.parameters.get("std", 1)
            return base_value + correlation_adjustment * std
        
        return base_value
    
    def _apply_constraints(self, value: Any, field_config: FieldSimulationConfig) -> Any:
        """Apply field constraints"""
        
        if value is None:
            return None
        
        constraints = field_config.constraints
        
        if isinstance(value, (int, float)):
            if "min" in constraints:
                value = max(value, constraints["min"])
            if "max" in constraints:
                value = min(value, constraints["max"])
        
        # Convert to appropriate type
        if field_config.field_type == "integer":
            return int(round(value)) if isinstance(value, (int, float)) else value
        elif field_config.field_type == "float":
            return float(value) if isinstance(value, (int, float)) else value
        elif field_config.field_type == "string":
            return str(value)
        elif field_config.field_type == "boolean":
            return bool(value)
        
        return value
    
    def _apply_quality_issues(
        self,
        record: Dict[str, Any],
        quality_issues: Dict[str, float]
    ) -> Dict[str, Any]:
        """Apply data quality issues to a record"""
        
        modified_record = record.copy()
        
        # Missing values
        missing_prob = quality_issues.get("missing_values", 0.0)
        if missing_prob > 0:
            for field_name in list(modified_record.keys()):
                if random.random() < missing_prob:
                    modified_record[field_name] = None
        
        # Outliers
        outlier_prob = quality_issues.get("outliers", 0.0)
        if outlier_prob > 0:
            for field_name, value in modified_record.items():
                if isinstance(value, (int, float)) and random.random() < outlier_prob:
                    # Generate outlier (10x normal value)
                    modified_record[field_name] = value * random.choice([10, -10, 0.1])
        
        # Inconsistent formats
        format_prob = quality_issues.get("inconsistent_formats", 0.0)
        if format_prob > 0:
            for field_name, value in modified_record.items():
                if isinstance(value, str) and random.random() < format_prob:
                    # Introduce format inconsistencies
                    if "@" in value:  # Email
                        modified_record[field_name] = value.replace("@", " at ")
                    elif "-" in value:  # ID with dashes
                        modified_record[field_name] = value.replace("-", "_")
        
        return modified_record
    
    def _calculate_quality_metrics(
        self,
        data: List[Dict[str, Any]],
        config: TemplateSimulationConfig
    ) -> Dict[str, Any]:
        """Calculate data quality metrics"""
        
        if not data:
            return {}
        
        total_records = len(data)
        total_fields = len(config.field_configs)
        
        # Count missing values
        missing_count = 0
        for record in data:
            for value in record.values():
                if value is None:
                    missing_count += 1
        
        missing_rate = missing_count / (total_records * total_fields) if total_records > 0 else 0
        
        # Count outliers (simplified)
        outlier_count = 0
        for record in data:
            for value in record.values():
                if isinstance(value, (int, float)) and abs(value) > 1000000:  # Simple outlier detection
                    outlier_count += 1
        
        outlier_rate = outlier_count / (total_records * total_fields) if total_records > 0 else 0
        
        # Calculate field completeness
        field_completeness = {}
        for field_config in config.field_configs:
            field_name = field_config.field_name
            non_null_count = sum(1 for record in data if record.get(field_name) is not None)
            field_completeness[field_name] = non_null_count / total_records if total_records > 0 else 0
        
        return {
            "total_records": total_records,
            "total_fields": total_fields,
            "missing_rate": missing_rate,
            "outlier_rate": outlier_rate,
            "field_completeness": field_completeness,
            "overall_quality_score": 1.0 - missing_rate - outlier_rate
        }
    
    def get_simulation_config(self, template_id: str) -> Optional[TemplateSimulationConfig]:
        """Get simulation configuration for a template"""
        return self.simulation_configs.get(template_id)
    
    def list_available_scenarios(self, template_id: str) -> List[str]:
        """List available scenarios for a template"""
        config = self.simulation_configs.get(template_id)
        if config:
            scenarios = ["default"] + list(config.scenarios.keys())
            return scenarios
        return []
    
    async def generate_batch_simulations(
        self,
        template_ids: List[str],
        record_count: int = 1000,
        scenarios: Optional[List[str]] = None
    ) -> Dict[str, List[SimulationResult]]:
        """Generate batch simulations for multiple templates"""
        
        results = {}
        
        for template_id in template_ids:
            template_results = []
            template_scenarios = scenarios or self.list_available_scenarios(template_id)
            
            for scenario in template_scenarios:
                try:
                    result = await self.simulate_data(template_id, record_count, scenario)
                    template_results.append(result)
                except Exception as e:
                    self.logger.error(f"Failed to simulate {template_id} scenario {scenario}: {e}")
            
            results[template_id] = template_results
        
        return results
    
    def export_simulation_data(
        self,
        simulation_result: SimulationResult,
        format: str = "json"
    ) -> Union[str, bytes]:
        """Export simulation data in various formats"""
        
        if format == "json":
            return json.dumps({
                "simulation_id": simulation_result.simulation_id,
                "template_id": simulation_result.template_id,
                "scenario": simulation_result.scenario,
                "metadata": simulation_result.metadata,
                "quality_metrics": simulation_result.quality_metrics,
                "data": simulation_result.data
            }, indent=2, default=str)
        
        elif format == "csv":
            import csv
            import io
            
            output = io.StringIO()
            if simulation_result.data:
                writer = csv.DictWriter(output, fieldnames=simulation_result.data[0].keys())
                writer.writeheader()
                writer.writerows(simulation_result.data)
            
            return output.getvalue()
        
        else:
            raise ValueError(f"Unsupported export format: {format}")
    
    async def validate_simulation_quality(
        self,
        simulation_result: SimulationResult,
        quality_thresholds: Optional[Dict[str, float]] = None
    ) -> Dict[str, Any]:
        """Validate simulation data quality"""
        
        default_thresholds = {
            "min_completeness": 0.95,
            "max_missing_rate": 0.05,
            "max_outlier_rate": 0.02,
            "min_quality_score": 0.9
        }
        
        thresholds = quality_thresholds or default_thresholds
        metrics = simulation_result.quality_metrics
        
        validation_results = {
            "passed": True,
            "issues": [],
            "recommendations": []
        }
        
        # Check completeness
        avg_completeness = sum(metrics["field_completeness"].values()) / len(metrics["field_completeness"])
        if avg_completeness < thresholds["min_completeness"]:
            validation_results["passed"] = False
            validation_results["issues"].append(f"Low average completeness: {avg_completeness:.3f}")
            validation_results["recommendations"].append("Reduce null_probability in field configurations")
        
        # Check missing rate
        if metrics["missing_rate"] > thresholds["max_missing_rate"]:
            validation_results["passed"] = False
            validation_results["issues"].append(f"High missing rate: {metrics['missing_rate']:.3f}")
        
        # Check outlier rate
        if metrics["outlier_rate"] > thresholds["max_outlier_rate"]:
            validation_results["passed"] = False
            validation_results["issues"].append(f"High outlier rate: {metrics['outlier_rate']:.3f}")
        
        # Check overall quality
        if metrics["overall_quality_score"] < thresholds["min_quality_score"]:
            validation_results["passed"] = False
            validation_results["issues"].append(f"Low quality score: {metrics['overall_quality_score']:.3f}")
        
        return validation_results
