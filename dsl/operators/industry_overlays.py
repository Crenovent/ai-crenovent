"""
Industry Overlays - Industry-specific ML models and configurations
Task 4.1.10: Add industry overlays with ML models (SaaS churn, Banking credit score, Insurance fraud)
"""

import asyncio
import json
import uuid
from typing import Dict, List, Any, Optional, Union
from dataclasses import dataclass, field
from datetime import datetime
import logging
from pathlib import Path

from .node_registry_service import MLModelConfig, NodeRegistryService

logger = logging.getLogger(__name__)

@dataclass
class IndustryOverlay:
    """Industry-specific overlay configuration"""
    industry: str
    overlay_name: str
    description: str
    ml_models: List[MLModelConfig]
    governance_rules: Dict[str, Any]
    compliance_requirements: List[str]
    data_retention_policies: Dict[str, Any]
    audit_requirements: Dict[str, Any]
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)

class IndustryOverlayService:
    """
    Service for managing industry-specific ML overlays
    
    Provides:
    - SaaS churn prediction models
    - Banking credit scoring models
    - Insurance fraud detection models
    - E-commerce recommendation models
    - Financial services risk assessment models
    - Industry-specific governance rules
    - Compliance requirements per industry
    """
    
    def __init__(self, node_registry: NodeRegistryService):
        self.node_registry = node_registry
        self.logger = logging.getLogger(__name__)
        self._overlays = {}
        self._initialize_industry_overlays()
    
    def _initialize_industry_overlays(self):
        """Initialize industry-specific overlays"""
        try:
            # SaaS Overlay
            saas_overlay = self._create_saas_overlay()
            self._overlays['saas'] = saas_overlay
            
            # Banking Overlay
            banking_overlay = self._create_banking_overlay()
            self._overlays['banking'] = banking_overlay
            
            # Insurance Overlay
            insurance_overlay = self._create_insurance_overlay()
            self._overlays['insurance'] = insurance_overlay
            
            # E-commerce Overlay
            ecommerce_overlay = self._create_ecommerce_overlay()
            self._overlays['ecommerce'] = ecommerce_overlay
            
            # Financial Services Overlay
            fs_overlay = self._create_financial_services_overlay()
            self._overlays['financial_services'] = fs_overlay
            
            self.logger.info(f"Initialized {len(self._overlays)} industry overlays")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize industry overlays: {e}")
            raise
    
    def _create_saas_overlay(self) -> IndustryOverlay:
        """Create SaaS industry overlay with churn prediction models"""
        
        # SaaS Churn Prediction Model
        churn_model = MLModelConfig(
            model_id="saas_churn_predictor_v2",
            model_name="SaaS Churn Predictor Pro",
            model_type="classification",
            model_version="2.0.0",
            industry="saas",
            description="Advanced churn prediction for SaaS companies with feature engineering",
            input_features=[
                "days_since_last_login",
                "monthly_recurring_revenue",
                "support_tickets_count",
                "feature_usage_score",
                "contract_length_months",
                "payment_history_score",
                "engagement_score",
                "product_usage_frequency",
                "customer_success_score",
                "renewal_probability"
            ],
            output_schema={
                "prediction": {"type": "string", "enum": ["low_risk", "medium_risk", "high_risk", "critical_risk"]},
                "confidence": {"type": "float", "min": 0.0, "max": 1.0},
                "probability": {"type": "object", "properties": {
                    "low_risk": {"type": "float"},
                    "medium_risk": {"type": "float"},
                    "high_risk": {"type": "float"},
                    "critical_risk": {"type": "float"}
                }},
                "churn_score": {"type": "float", "min": 0.0, "max": 100.0},
                "retention_recommendations": {"type": "array", "items": {"type": "string"}}
            },
            confidence_threshold=0.8,
            tags=["churn", "saas", "retention", "customer_success", "revenue_ops"]
        )
        
        # SaaS Feature Usage Analyzer
        usage_model = MLModelConfig(
            model_id="saas_usage_analyzer_v1",
            model_name="SaaS Feature Usage Analyzer",
            model_type="regression",
            model_version="1.0.0",
            industry="saas",
            description="Analyzes feature usage patterns to predict adoption and expansion",
            input_features=[
                "feature_adoption_rate",
                "feature_usage_depth",
                "user_engagement_score",
                "feature_complexity_score",
                "training_completion_rate",
                "support_interaction_frequency"
            ],
            output_schema={
                "usage_score": {"type": "float", "min": 0.0, "max": 100.0},
                "adoption_probability": {"type": "float", "min": 0.0, "max": 1.0},
                "expansion_opportunity": {"type": "string", "enum": ["low", "medium", "high"]},
                "recommended_features": {"type": "array", "items": {"type": "string"}}
            },
            confidence_threshold=0.75,
            tags=["usage", "adoption", "expansion", "product_analytics"]
        )
        
        return IndustryOverlay(
            industry="saas",
            overlay_name="SaaS Revenue Operations",
            description="Comprehensive SaaS overlay with churn prediction and usage analytics",
            ml_models=[churn_model, usage_model],
            governance_rules={
                "data_retention_days": 365,
                "pii_handling": "encrypted",
                "audit_frequency": "monthly",
                "model_retraining": "quarterly",
                "confidence_thresholds": {
                    "churn_prediction": 0.8,
                    "usage_analysis": 0.75
                }
            },
            compliance_requirements=[
                "GDPR",
                "CCPA",
                "SOC2",
                "ISO27001"
            ],
            data_retention_policies={
                "customer_data": "2_years",
                "usage_data": "1_year",
                "prediction_logs": "6_months",
                "audit_logs": "7_years"
            },
            audit_requirements={
                "model_performance_review": "monthly",
                "bias_audit": "quarterly",
                "data_quality_check": "weekly",
                "compliance_audit": "annually"
            }
        )
    
    def _create_banking_overlay(self) -> IndustryOverlay:
        """Create Banking industry overlay with credit scoring models"""
        
        # Banking Credit Scorer
        credit_model = MLModelConfig(
            model_id="banking_credit_scorer_v2",
            model_name="Banking Credit Scorer Pro",
            model_type="regression",
            model_version="2.0.0",
            industry="banking",
            description="Advanced credit scoring for banking applications with regulatory compliance",
            input_features=[
                "credit_score",
                "annual_income",
                "debt_to_income_ratio",
                "employment_length_years",
                "loan_amount",
                "collateral_value",
                "credit_history_length",
                "recent_inquiries",
                "payment_history_score",
                "credit_utilization_ratio"
            ],
            output_schema={
                "credit_score": {"type": "float", "min": 300, "max": 850},
                "confidence": {"type": "float", "min": 0.0, "max": 1.0},
                "risk_tier": {"type": "string", "enum": ["excellent", "good", "fair", "poor", "very_poor"]},
                "approval_probability": {"type": "float", "min": 0.0, "max": 1.0},
                "interest_rate_range": {"type": "object", "properties": {
                    "min_rate": {"type": "float"},
                    "max_rate": {"type": "float"}
                }},
                "regulatory_flags": {"type": "array", "items": {"type": "string"}}
            },
            confidence_threshold=0.85,
            tags=["credit", "banking", "risk", "scoring", "compliance"]
        )
        
        # Banking Fraud Detector
        fraud_model = MLModelConfig(
            model_id="banking_fraud_detector_v1",
            model_name="Banking Fraud Detector",
            model_type="anomaly_detection",
            model_version="1.0.0",
            industry="banking",
            description="Real-time fraud detection for banking transactions",
            input_features=[
                "transaction_amount",
                "transaction_frequency",
                "geographic_risk_score",
                "time_of_day",
                "merchant_category_risk",
                "device_fingerprint_risk",
                "velocity_risk_score",
                "pattern_anomaly_score"
            ],
            output_schema={
                "fraud_score": {"type": "float", "min": 0.0, "max": 100.0},
                "is_fraudulent": {"type": "boolean"},
                "confidence": {"type": "float", "min": 0.0, "max": 1.0},
                "risk_factors": {"type": "array", "items": {"type": "string"}},
                "recommended_action": {"type": "string", "enum": ["approve", "review", "decline", "escalate"]}
            },
            confidence_threshold=0.9,
            tags=["fraud", "banking", "anomaly", "detection", "security"]
        )
        
        return IndustryOverlay(
            industry="banking",
            overlay_name="Banking Risk Management",
            description="Comprehensive banking overlay with credit scoring and fraud detection",
            ml_models=[credit_model, fraud_model],
            governance_rules={
                "data_retention_days": 2555,  # 7 years
                "pii_handling": "encrypted_plus_tokenized",
                "audit_frequency": "daily",
                "model_retraining": "monthly",
                "confidence_thresholds": {
                    "credit_scoring": 0.85,
                    "fraud_detection": 0.9
                },
                "regulatory_approval_required": True
            },
            compliance_requirements=[
                "Basel_III",
                "PCI_DSS",
                "SOX",
                "FFIEC",
                "GDPR",
                "CCPA"
            ],
            data_retention_policies={
                "credit_data": "7_years",
                "transaction_data": "5_years",
                "fraud_logs": "10_years",
                "audit_logs": "7_years"
            },
            audit_requirements={
                "model_performance_review": "daily",
                "bias_audit": "monthly",
                "data_quality_check": "daily",
                "compliance_audit": "quarterly",
                "regulatory_reporting": "monthly"
            }
        )
    
    def _create_insurance_overlay(self) -> IndustryOverlay:
        """Create Insurance industry overlay with fraud detection models"""
        
        # Insurance Fraud Detector
        fraud_model = MLModelConfig(
            model_id="insurance_fraud_detector_v2",
            model_name="Insurance Fraud Detector Pro",
            model_type="anomaly_detection",
            model_version="2.0.0",
            industry="insurance",
            description="Advanced fraud detection for insurance claims with regulatory compliance",
            input_features=[
                "claim_amount",
                "policy_age_days",
                "claim_frequency",
                "geographic_risk_score",
                "time_to_report_hours",
                "previous_claims_count",
                "claim_type_risk_score",
                "policyholder_risk_score",
                "third_party_involvement",
                "medical_provider_risk_score"
            ],
            output_schema={
                "fraud_score": {"type": "float", "min": 0.0, "max": 100.0},
                "is_fraudulent": {"type": "boolean"},
                "confidence": {"type": "float", "min": 0.0, "max": 1.0},
                "risk_factors": {"type": "array", "items": {"type": "string"}},
                "investigation_priority": {"type": "string", "enum": ["low", "medium", "high", "urgent"]},
                "regulatory_flags": {"type": "array", "items": {"type": "string"}}
            },
            confidence_threshold=0.88,
            tags=["fraud", "insurance", "anomaly", "detection", "claims"]
        )
        
        # Insurance Risk Assessor
        risk_model = MLModelConfig(
            model_id="insurance_risk_assessor_v1",
            model_name="Insurance Risk Assessor",
            model_type="regression",
            model_version="1.0.0",
            industry="insurance",
            description="Comprehensive risk assessment for insurance underwriting",
            input_features=[
                "age",
                "health_score",
                "lifestyle_risk_factors",
                "occupation_risk_score",
                "geographic_risk_score",
                "family_history_score",
                "previous_claims_history",
                "policy_type_risk",
                "coverage_amount",
                "deductible_amount"
            ],
            output_schema={
                "risk_score": {"type": "float", "min": 0.0, "max": 100.0},
                "premium_multiplier": {"type": "float", "min": 0.5, "max": 3.0},
                "approval_probability": {"type": "float", "min": 0.0, "max": 1.0},
                "risk_category": {"type": "string", "enum": ["low", "medium", "high", "very_high"]},
                "underwriting_recommendations": {"type": "array", "items": {"type": "string"}}
            },
            confidence_threshold=0.8,
            tags=["risk", "insurance", "underwriting", "pricing"]
        )
        
        return IndustryOverlay(
            industry="insurance",
            overlay_name="Insurance Risk Management",
            description="Comprehensive insurance overlay with fraud detection and risk assessment",
            ml_models=[fraud_model, risk_model],
            governance_rules={
                "data_retention_days": 2555,  # 7 years
                "pii_handling": "encrypted_plus_tokenized",
                "audit_frequency": "daily",
                "model_retraining": "monthly",
                "confidence_thresholds": {
                    "fraud_detection": 0.88,
                    "risk_assessment": 0.8
                },
                "regulatory_approval_required": True
            },
            compliance_requirements=[
                "HIPAA",
                "SOX",
                "NAIC",
                "GDPR",
                "CCPA",
                "State_Insurance_Regulations"
            ],
            data_retention_policies={
                "claim_data": "7_years",
                "policy_data": "7_years",
                "fraud_logs": "10_years",
                "audit_logs": "7_years"
            },
            audit_requirements={
                "model_performance_review": "daily",
                "bias_audit": "monthly",
                "data_quality_check": "daily",
                "compliance_audit": "quarterly",
                "regulatory_reporting": "monthly"
            }
        )
    
    def _create_ecommerce_overlay(self) -> IndustryOverlay:
        """Create E-commerce industry overlay with recommendation models"""
        
        # E-commerce Recommender
        recommender_model = MLModelConfig(
            model_id="ecommerce_recommender_v2",
            model_name="E-commerce Recommender Pro",
            model_type="classification",
            model_version="2.0.0",
            industry="ecommerce",
            description="Advanced product recommendation system for e-commerce",
            input_features=[
                "user_id",
                "product_category",
                "purchase_history_score",
                "browsing_behavior_score",
                "seasonal_factor",
                "price_sensitivity",
                "brand_preference_score",
                "product_rating",
                "inventory_level",
                "promotional_factor"
            ],
            output_schema={
                "recommended_products": {"type": "array", "items": {"type": "string"}},
                "confidence": {"type": "float", "min": 0.0, "max": 1.0},
                "recommendation_reason": {"type": "string"},
                "expected_conversion_rate": {"type": "float", "min": 0.0, "max": 1.0},
                "personalization_score": {"type": "float", "min": 0.0, "max": 100.0}
            },
            confidence_threshold=0.7,
            tags=["recommendation", "ecommerce", "personalization", "conversion"]
        )
        
        # E-commerce Price Optimizer
        pricing_model = MLModelConfig(
            model_id="ecommerce_price_optimizer_v1",
            model_name="E-commerce Price Optimizer",
            model_type="regression",
            model_version="1.0.0",
            industry="ecommerce",
            description="Dynamic pricing optimization for e-commerce products",
            input_features=[
                "product_id",
                "current_price",
                "competitor_prices",
                "demand_elasticity",
                "inventory_level",
                "seasonal_demand_factor",
                "promotional_history",
                "customer_segment",
                "market_conditions",
                "product_lifecycle_stage"
            ],
            output_schema={
                "optimal_price": {"type": "float", "min": 0.0},
                "confidence": {"type": "float", "min": 0.0, "max": 1.0},
                "expected_revenue_impact": {"type": "float"},
                "price_elasticity": {"type": "float"},
                "pricing_strategy": {"type": "string", "enum": ["premium", "competitive", "discount", "penetration"]}
            },
            confidence_threshold=0.75,
            tags=["pricing", "optimization", "revenue", "ecommerce"]
        )
        
        return IndustryOverlay(
            industry="ecommerce",
            overlay_name="E-commerce Intelligence",
            description="Comprehensive e-commerce overlay with recommendation and pricing optimization",
            ml_models=[recommender_model, pricing_model],
            governance_rules={
                "data_retention_days": 730,  # 2 years
                "pii_handling": "encrypted",
                "audit_frequency": "weekly",
                "model_retraining": "weekly",
                "confidence_thresholds": {
                    "recommendations": 0.7,
                    "pricing": 0.75
                }
            },
            compliance_requirements=[
                "GDPR",
                "CCPA",
                "PCI_DSS",
                "COPPA"
            ],
            data_retention_policies={
                "user_data": "2_years",
                "transaction_data": "2_years",
                "recommendation_logs": "1_year",
                "audit_logs": "3_years"
            },
            audit_requirements={
                "model_performance_review": "weekly",
                "bias_audit": "monthly",
                "data_quality_check": "daily",
                "compliance_audit": "quarterly"
            }
        )
    
    def _create_financial_services_overlay(self) -> IndustryOverlay:
        """Create Financial Services industry overlay with risk assessment models"""
        
        # Financial Services Risk Assessor
        risk_model = MLModelConfig(
            model_id="fs_risk_assessor_v2",
            model_name="Financial Services Risk Assessor Pro",
            model_type="regression",
            model_version="2.0.0",
            industry="financial_services",
            description="Advanced risk assessment for financial services with regulatory compliance",
            input_features=[
                "transaction_amount",
                "customer_risk_score",
                "transaction_frequency",
                "geographic_risk",
                "time_of_day",
                "merchant_category_risk",
                "account_age_days",
                "credit_score",
                "income_verification_score",
                "regulatory_risk_factors"
            ],
            output_schema={
                "risk_score": {"type": "float", "min": 0.0, "max": 100.0},
                "confidence": {"type": "float", "min": 0.0, "max": 1.0},
                "action_required": {"type": "string", "enum": ["approve", "review", "decline", "escalate"]},
                "regulatory_flags": {"type": "array", "items": {"type": "string"}},
                "compliance_risk_level": {"type": "string", "enum": ["low", "medium", "high", "critical"]},
                "monitoring_recommendations": {"type": "array", "items": {"type": "string"}}
            },
            confidence_threshold=0.85,
            tags=["risk", "financial_services", "compliance", "transaction", "regulatory"]
        )
        
        # Financial Services Market Risk Analyzer
        market_model = MLModelConfig(
            model_id="fs_market_risk_analyzer_v1",
            model_name="Financial Services Market Risk Analyzer",
            model_type="regression",
            model_version="1.0.0",
            industry="financial_services",
            description="Market risk analysis for financial services portfolios",
            input_features=[
                "portfolio_value",
                "asset_allocation",
                "market_volatility",
                "interest_rate_risk",
                "currency_risk",
                "sector_concentration",
                "liquidity_risk",
                "credit_risk",
                "operational_risk",
                "regulatory_risk"
            ],
            output_schema={
                "market_risk_score": {"type": "float", "min": 0.0, "max": 100.0},
                "var_95": {"type": "float"},
                "expected_shortfall": {"type": "float"},
                "risk_category": {"type": "string", "enum": ["low", "medium", "high", "very_high"]},
                "hedging_recommendations": {"type": "array", "items": {"type": "string"}}
            },
            confidence_threshold=0.8,
            tags=["market_risk", "portfolio", "var", "hedging", "financial_services"]
        )
        
        return IndustryOverlay(
            industry="financial_services",
            overlay_name="Financial Services Risk Management",
            description="Comprehensive financial services overlay with risk assessment and market analysis",
            ml_models=[risk_model, market_model],
            governance_rules={
                "data_retention_days": 2555,  # 7 years
                "pii_handling": "encrypted_plus_tokenized",
                "audit_frequency": "daily",
                "model_retraining": "weekly",
                "confidence_thresholds": {
                    "risk_assessment": 0.85,
                    "market_analysis": 0.8
                },
                "regulatory_approval_required": True
            },
            compliance_requirements=[
                "Basel_III",
                "MiFID_II",
                "Dodd_Frank",
                "SOX",
                "GDPR",
                "CCPA"
            ],
            data_retention_policies={
                "transaction_data": "7_years",
                "risk_data": "7_years",
                "market_data": "5_years",
                "audit_logs": "7_years"
            },
            audit_requirements={
                "model_performance_review": "daily",
                "bias_audit": "weekly",
                "data_quality_check": "daily",
                "compliance_audit": "monthly",
                "regulatory_reporting": "daily"
            }
        )
    
    async def get_overlay(self, industry: str) -> Optional[IndustryOverlay]:
        """Get industry overlay by industry name"""
        return self._overlays.get(industry.lower())
    
    async def list_overlays(self) -> List[IndustryOverlay]:
        """List all available industry overlays"""
        return list(self._overlays.values())
    
    async def register_overlay_models(self, industry: str) -> bool:
        """Register all models from an industry overlay to the node registry"""
        try:
            overlay = await self.get_overlay(industry)
            if not overlay:
                self.logger.error(f"Overlay not found for industry: {industry}")
                return False
            
            success_count = 0
            for model in overlay.ml_models:
                success = await self.node_registry.register_model(model)
                if success:
                    success_count += 1
                else:
                    self.logger.warning(f"Failed to register model: {model.model_id}")
            
            self.logger.info(f"Registered {success_count}/{len(overlay.ml_models)} models for {industry} overlay")
            return success_count > 0
            
        except Exception as e:
            self.logger.error(f"Failed to register overlay models for {industry}: {e}")
            return False
    
    async def register_all_overlay_models(self) -> Dict[str, bool]:
        """Register all models from all industry overlays"""
        results = {}
        
        for industry in self._overlays.keys():
            results[industry] = await self.register_overlay_models(industry)
        
        return results
    
    async def get_industry_governance_rules(self, industry: str) -> Optional[Dict[str, Any]]:
        """Get governance rules for a specific industry"""
        overlay = await self.get_overlay(industry)
        return overlay.governance_rules if overlay else None
    
    async def get_industry_compliance_requirements(self, industry: str) -> Optional[List[str]]:
        """Get compliance requirements for a specific industry"""
        overlay = await self.get_overlay(industry)
        return overlay.compliance_requirements if overlay else None
    
    async def get_industry_audit_requirements(self, industry: str) -> Optional[Dict[str, Any]]:
        """Get audit requirements for a specific industry"""
        overlay = await self.get_overlay(industry)
        return overlay.audit_requirements if overlay else None
    
    async def validate_model_compliance(
        self, 
        model_id: str, 
        industry: str
    ) -> Dict[str, Any]:
        """Validate model compliance with industry requirements"""
        try:
            overlay = await self.get_overlay(industry)
            if not overlay:
                return {
                    "compliant": False,
                    "errors": [f"No overlay found for industry: {industry}"]
                }
            
            # Get model from registry
            model = await self.node_registry.get_model(model_id)
            if not model:
                return {
                    "compliant": False,
                    "errors": [f"Model not found: {model_id}"]
                }
            
            errors = []
            warnings = []
            
            # Check confidence threshold
            governance_rules = overlay.governance_rules
            if 'confidence_thresholds' in governance_rules:
                model_type = model.model_type
                if model_type in governance_rules['confidence_thresholds']:
                    required_threshold = governance_rules['confidence_thresholds'][model_type]
                    if model.confidence_threshold < required_threshold:
                        errors.append(f"Confidence threshold {model.confidence_threshold} below required {required_threshold}")
            
            # Check explainability
            if not model.explainability_enabled:
                warnings.append("Explainability not enabled - may be required for compliance")
            
            # Check fallback
            if not model.fallback_enabled:
                warnings.append("Fallback not enabled - may be required for risk management")
            
            # Check tags alignment
            industry_tags = set()
            for overlay_model in overlay.ml_models:
                industry_tags.update(overlay_model.tags)
            
            model_tags = set(model.tags)
            if not model_tags.intersection(industry_tags):
                warnings.append("Model tags don't align with industry requirements")
            
            return {
                "compliant": len(errors) == 0,
                "errors": errors,
                "warnings": warnings,
                "industry": industry,
                "model_id": model_id
            }
            
        except Exception as e:
            self.logger.error(f"Failed to validate model compliance: {e}")
            return {
                "compliant": False,
                "errors": [f"Validation failed: {e}"]
            }
    
    async def get_industry_ml_models(self, industry: str) -> List[MLModelConfig]:
        """Get all ML models for a specific industry"""
        overlay = await self.get_overlay(industry)
        return overlay.ml_models if overlay else []
    
    async def create_custom_industry_overlay(
        self,
        industry: str,
        overlay_name: str,
        description: str,
        ml_models: List[MLModelConfig],
        governance_rules: Dict[str, Any],
        compliance_requirements: List[str],
        data_retention_policies: Dict[str, Any],
        audit_requirements: Dict[str, Any]
    ) -> bool:
        """Create a custom industry overlay"""
        try:
            overlay = IndustryOverlay(
                industry=industry,
                overlay_name=overlay_name,
                description=description,
                ml_models=ml_models,
                governance_rules=governance_rules,
                compliance_requirements=compliance_requirements,
                data_retention_policies=data_retention_policies,
                audit_requirements=audit_requirements
            )
            
            self._overlays[industry.lower()] = overlay
            
            # Register models
            await self.register_overlay_models(industry)
            
            self.logger.info(f"Created custom overlay for industry: {industry}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to create custom overlay for {industry}: {e}")
            return False
