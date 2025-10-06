"""
SaaS Intent Taxonomy - Chapter 15.1.4 Implementation
===================================================
Task 15.1.4: Define SaaS intent taxonomies (ARR, churn, comp-plan, QBR)

Implements comprehensive SaaS-specific intent classification with:
- ARR analysis and growth tracking intents
- Churn prediction and retention intents  
- Compensation plan and quota management intents
- QBR and account planning intents
- Dynamic parameter extraction
- Confidence scoring and validation
"""

import logging
import re
import json
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict
from enum import Enum
from datetime import datetime, timezone
import uuid

logger = logging.getLogger(__name__)

class SaaSIntentCategory(Enum):
    """SaaS-specific intent categories"""
    ARR_ANALYSIS = "arr_analysis"
    CHURN_PREDICTION = "churn_prediction" 
    COMP_PLAN = "comp_plan"
    QBR = "qbr"
    PIPELINE_HYGIENE = "pipeline_hygiene"
    FORECAST_ACCURACY = "forecast_accuracy"
    CUSTOMER_SUCCESS = "customer_success"
    SUBSCRIPTION_LIFECYCLE = "subscription_lifecycle"
    REVENUE_RECOGNITION = "revenue_recognition"
    USAGE_ANALYTICS = "usage_analytics"

class AutomationRoute(Enum):
    """Automation routing options"""
    RBA = "RBA"      # Rule-Based Automation
    RBIA = "RBIA"    # Rule-Based Intelligent Automation  
    AALA = "AALA"    # AI Agent-Led Automation

@dataclass
class SaaSIntentPattern:
    """SaaS intent pattern definition"""
    pattern_id: str
    category: SaaSIntentCategory
    pattern_name: str
    regex_patterns: List[str]
    keywords: List[str]
    automation_route: AutomationRoute
    confidence_weight: float
    parameter_extractors: Dict[str, str]
    business_context: Dict[str, Any]
    compliance_requirements: List[str]

@dataclass
class ExtractedParameters:
    """Extracted parameters from intent"""
    time_period: Optional[str] = None
    metric_type: Optional[str] = None
    account_name: Optional[str] = None
    segment: Optional[str] = None
    threshold: Optional[float] = None
    currency: Optional[str] = None
    region: Optional[str] = None
    product: Optional[str] = None
    team: Optional[str] = None
    custom_parameters: Dict[str, Any] = None

    def __post_init__(self):
        if self.custom_parameters is None:
            self.custom_parameters = {}

@dataclass
class SaaSIntentClassification:
    """Result of SaaS intent classification"""
    intent_id: str
    category: SaaSIntentCategory
    automation_route: AutomationRoute
    confidence_score: float
    extracted_parameters: ExtractedParameters
    matched_patterns: List[str]
    business_context: Dict[str, Any]
    compliance_flags: List[str]
    recommended_capability: Optional[str] = None

class SaaSIntentTaxonomy:
    """
    SaaS Intent Taxonomy Engine
    Task 15.1.4: Define SaaS intent taxonomies
    """
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.intent_patterns = self._initialize_saas_patterns()
        self.parameter_extractors = self._initialize_parameter_extractors()
        
    def _initialize_saas_patterns(self) -> List[SaaSIntentPattern]:
        """Initialize SaaS-specific intent patterns"""
        return [
            # ARR Analysis Patterns
            SaaSIntentPattern(
                pattern_id="saas_arr_001",
                category=SaaSIntentCategory.ARR_ANALYSIS,
                pattern_name="ARR Growth Analysis",
                regex_patterns=[
                    r"(?i).*\b(arr|annual recurring revenue)\b.*\b(growth|trend|analysis|forecast)\b.*",
                    r"(?i).*\b(analyze|review|track)\b.*\b(arr|revenue)\b.*",
                    r"(?i).*\b(revenue)\b.*\b(growth|expansion|contraction)\b.*"
                ],
                keywords=["arr", "annual recurring revenue", "revenue growth", "mrr", "monthly recurring"],
                automation_route=AutomationRoute.RBA,
                confidence_weight=0.9,
                parameter_extractors={
                    "time_period": r"(?i)\b(q[1-4]|quarter|month|year|ytd|mtd)\s*(\d{4})?\b",
                    "metric_type": r"(?i)\b(growth|expansion|contraction|churn|net retention)\b",
                    "segment": r"(?i)\b(enterprise|mid-market|smb|startup)\b"
                },
                business_context={
                    "business_impact": "high",
                    "stakeholders": ["CRO", "CFO", "RevOps"],
                    "typical_frequency": "monthly",
                    "data_sources": ["subscription_data", "billing_data", "crm_data"]
                },
                compliance_requirements=["SOX"]
            ),
            
            SaaSIntentPattern(
                pattern_id="saas_arr_002", 
                category=SaaSIntentCategory.ARR_ANALYSIS,
                pattern_name="ARR Variance Detection",
                regex_patterns=[
                    r"(?i).*\b(arr|revenue)\b.*\b(variance|deviation|anomaly|unusual)\b.*",
                    r"(?i).*\b(why|what caused|explain)\b.*\b(arr|revenue)\b.*\b(drop|increase|change)\b.*"
                ],
                keywords=["variance", "anomaly", "deviation", "unusual revenue"],
                automation_route=AutomationRoute.RBIA,
                confidence_weight=0.85,
                parameter_extractors={
                    "time_period": r"(?i)\b(last|previous|current)\s*(week|month|quarter|year)\b",
                    "threshold": r"(?i)\b(\d+(?:\.\d+)?)\s*(%|percent|dollar|usd)\b"
                },
                business_context={
                    "business_impact": "critical",
                    "stakeholders": ["CRO", "RevOps", "Finance"],
                    "urgency": "high",
                    "requires_investigation": True
                },
                compliance_requirements=["SOX", "GDPR"]
            ),
            
            # Churn Prediction Patterns
            SaaSIntentPattern(
                pattern_id="saas_churn_001",
                category=SaaSIntentCategory.CHURN_PREDICTION,
                pattern_name="Churn Risk Analysis",
                regex_patterns=[
                    r"(?i).*\b(churn|attrition)\b.*\b(risk|prediction|forecast|likely)\b.*",
                    r"(?i).*\b(predict|identify|find)\b.*\b(churn|at-risk|leaving)\b.*\b(customers|accounts)\b.*",
                    r"(?i).*\b(customer|account)\b.*\b(health|retention|renewal)\b.*"
                ],
                keywords=["churn", "attrition", "retention", "customer health", "renewal risk"],
                automation_route=AutomationRoute.RBIA,
                confidence_weight=0.88,
                parameter_extractors={
                    "time_period": r"(?i)\b(next|upcoming)\s*(\d+)?\s*(days?|weeks?|months?)\b",
                    "segment": r"(?i)\b(enterprise|mid-market|smb|trial|freemium)\b",
                    "threshold": r"(?i)\b(above|over|greater than)\s*(\d+(?:\.\d+)?)\s*(%|percent)\b"
                },
                business_context={
                    "business_impact": "high",
                    "stakeholders": ["Customer Success", "RevOps", "Sales"],
                    "actionable": True,
                    "intervention_required": True
                },
                compliance_requirements=["GDPR"]
            ),
            
            SaaSIntentPattern(
                pattern_id="saas_churn_002",
                category=SaaSIntentCategory.CHURN_PREDICTION,
                pattern_name="Customer Health Scoring",
                regex_patterns=[
                    r"(?i).*\b(customer|account)\b.*\b(health|score|rating)\b.*",
                    r"(?i).*\b(health)\b.*\b(dashboard|report|analysis)\b.*",
                    r"(?i).*\b(engagement|usage|adoption)\b.*\b(score|metrics)\b.*"
                ],
                keywords=["customer health", "health score", "engagement score", "adoption metrics"],
                automation_route=AutomationRoute.RBA,
                confidence_weight=0.82,
                parameter_extractors={
                    "account_name": r"(?i)\b(for|of)\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\b",
                    "metric_type": r"(?i)\b(usage|engagement|support|billing|feature adoption)\b"
                },
                business_context={
                    "business_impact": "medium",
                    "stakeholders": ["Customer Success", "Account Management"],
                    "frequency": "weekly",
                    "proactive": True
                },
                compliance_requirements=["GDPR"]
            ),
            
            # Compensation Plan Patterns
            SaaSIntentPattern(
                pattern_id="saas_comp_001",
                category=SaaSIntentCategory.COMP_PLAN,
                pattern_name="Compensation Plan Generation",
                regex_patterns=[
                    r"(?i).*\b(compensation|comp)\b.*\b(plan|structure|model)\b.*",
                    r"(?i).*\b(quota|target|commission)\b.*\b(plan|allocation|structure)\b.*",
                    r"(?i).*\b(sales)\b.*\b(compensation|incentive|bonus)\b.*"
                ],
                keywords=["compensation plan", "quota allocation", "commission structure", "sales incentive"],
                automation_route=AutomationRoute.RBA,
                confidence_weight=0.9,
                parameter_extractors={
                    "time_period": r"(?i)\b(fy|fiscal year|calendar year|q[1-4])\s*(\d{4})?\b",
                    "team": r"(?i)\b(sales|ae|sdr|bdr|customer success|account management)\b",
                    "region": r"(?i)\b(north america|emea|apac|americas|europe|asia)\b"
                },
                business_context={
                    "business_impact": "high",
                    "stakeholders": ["Sales Leadership", "Finance", "HR"],
                    "seasonal": True,
                    "approval_required": True
                },
                compliance_requirements=["SOX"]
            ),
            
            # QBR Patterns
            SaaSIntentPattern(
                pattern_id="saas_qbr_001",
                category=SaaSIntentCategory.QBR,
                pattern_name="QBR Preparation",
                regex_patterns=[
                    r"(?i).*\b(qbr|quarterly business review)\b.*",
                    r"(?i).*\b(quarterly|q[1-4])\b.*\b(review|meeting|presentation)\b.*",
                    r"(?i).*\b(account)\b.*\b(review|planning|strategy)\b.*"
                ],
                keywords=["qbr", "quarterly business review", "account review", "quarterly planning"],
                automation_route=AutomationRoute.AALA,
                confidence_weight=0.85,
                parameter_extractors={
                    "account_name": r"(?i)\b(for|with|account)\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\b",
                    "time_period": r"(?i)\b(q[1-4]|quarter)\s*(\d{4})?\b"
                },
                business_context={
                    "business_impact": "high",
                    "stakeholders": ["Account Executive", "Customer Success", "Sales Leadership"],
                    "preparation_time": "high",
                    "strategic": True
                },
                compliance_requirements=["GDPR"]
            ),
            
            # Pipeline Hygiene Patterns
            SaaSIntentPattern(
                pattern_id="saas_pipeline_001",
                category=SaaSIntentCategory.PIPELINE_HYGIENE,
                pattern_name="Pipeline Cleanup",
                regex_patterns=[
                    r"(?i).*\b(pipeline|deals?)\b.*\b(hygiene|cleanup|clean|stale)\b.*",
                    r"(?i).*\b(old|stale|stuck)\b.*\b(deals?|opportunities)\b.*",
                    r"(?i).*\b(pipeline)\b.*\b(review|audit|maintenance)\b.*"
                ],
                keywords=["pipeline hygiene", "stale deals", "pipeline cleanup", "deal review"],
                automation_route=AutomationRoute.RBA,
                confidence_weight=0.88,
                parameter_extractors={
                    "time_period": r"(?i)\b(older than|more than)\s*(\d+)\s*(days?|weeks?|months?)\b",
                    "stage": r"(?i)\b(discovery|qualification|proposal|negotiation|closed)\b"
                },
                business_context={
                    "business_impact": "medium",
                    "stakeholders": ["Sales Manager", "RevOps"],
                    "frequency": "weekly",
                    "maintenance": True
                },
                compliance_requirements=[]
            ),
            
            # Forecast Accuracy Patterns
            SaaSIntentPattern(
                pattern_id="saas_forecast_001",
                category=SaaSIntentCategory.FORECAST_ACCURACY,
                pattern_name="Forecast Accuracy Analysis",
                regex_patterns=[
                    r"(?i).*\b(forecast)\b.*\b(accuracy|precision|variance|error)\b.*",
                    r"(?i).*\b(forecast)\b.*\b(vs|versus|compared to)\b.*\b(actual|results)\b.*",
                    r"(?i).*\b(prediction|forecast)\b.*\b(performance|quality)\b.*"
                ],
                keywords=["forecast accuracy", "forecast variance", "prediction error", "forecast quality"],
                automation_route=AutomationRoute.RBIA,
                confidence_weight=0.87,
                parameter_extractors={
                    "time_period": r"(?i)\b(last|previous|current)\s*(quarter|month|year)\b",
                    "metric_type": r"(?i)\b(revenue|bookings|arr|pipeline)\b"
                },
                business_context={
                    "business_impact": "high",
                    "stakeholders": ["CRO", "Sales Leadership", "RevOps"],
                    "analytical": True,
                    "improvement_focused": True
                },
                compliance_requirements=["SOX"]
            )
        ]
    
    def _initialize_parameter_extractors(self) -> Dict[str, Any]:
        """Initialize parameter extraction patterns"""
        return {
            "time_extractors": {
                "quarter": r"(?i)\b(q[1-4]|quarter\s*[1-4])\s*(\d{4})?\b",
                "month": r"(?i)\b(january|february|march|april|may|june|july|august|september|october|november|december|jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)\s*(\d{4})?\b",
                "year": r"(?i)\b(fy|fiscal year|calendar year|cy)\s*(\d{4})\b",
                "relative": r"(?i)\b(last|previous|current|next|upcoming)\s*(week|month|quarter|year)\b",
                "range": r"(?i)\b(\d{1,2}\/\d{1,2}\/\d{4})\s*(?:to|through|-)\s*(\d{1,2}\/\d{1,2}\/\d{4})\b"
            },
            "metric_extractors": {
                "currency": r"(?i)\b(\$|usd|eur|gbp|dollar|euro|pound)\s*(\d+(?:,\d{3})*(?:\.\d{2})?)\b",
                "percentage": r"(?i)\b(\d+(?:\.\d+)?)\s*(%|percent)\b",
                "number": r"(?i)\b(\d+(?:,\d{3})*(?:\.\d+)?)\b"
            },
            "entity_extractors": {
                "account": r"(?i)\b(?:account|customer|client)\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\b",
                "product": r"(?i)\b(?:product|solution|platform)\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\b",
                "team": r"(?i)\b(?:team|group|division)\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\b"
            }
        }
    
    async def classify_intent(self, raw_input: str, context: Dict[str, Any] = None) -> SaaSIntentClassification:
        """
        Classify SaaS intent from raw input
        Task 15.1.4: SaaS intent classification with confidence scoring
        """
        try:
            context = context or {}
            intent_id = str(uuid.uuid4())
            
            # Find matching patterns
            matched_patterns = []
            best_match = None
            highest_confidence = 0.0
            
            for pattern in self.intent_patterns:
                confidence = await self._calculate_pattern_confidence(raw_input, pattern)
                if confidence > 0.3:  # Minimum confidence threshold
                    matched_patterns.append({
                        "pattern_id": pattern.pattern_id,
                        "confidence": confidence,
                        "category": pattern.category.value
                    })
                    
                    if confidence > highest_confidence:
                        highest_confidence = confidence
                        best_match = pattern
            
            if not best_match:
                # Fallback to general classification
                return await self._create_fallback_classification(intent_id, raw_input, context)
            
            # Extract parameters
            extracted_params = await self._extract_parameters(raw_input, best_match)
            
            # Determine compliance flags
            compliance_flags = await self._determine_compliance_flags(
                best_match.category, extracted_params, context
            )
            
            # Get recommended capability
            recommended_capability = await self._get_recommended_capability(
                best_match.category, best_match.automation_route, context
            )
            
            classification = SaaSIntentClassification(
                intent_id=intent_id,
                category=best_match.category,
                automation_route=best_match.automation_route,
                confidence_score=highest_confidence,
                extracted_parameters=extracted_params,
                matched_patterns=[p["pattern_id"] for p in matched_patterns],
                business_context=best_match.business_context,
                compliance_flags=compliance_flags,
                recommended_capability=recommended_capability
            )
            
            self.logger.info(f"✅ Classified SaaS intent: {best_match.category.value} (confidence: {highest_confidence:.3f})")
            return classification
            
        except Exception as e:
            self.logger.error(f"❌ Failed to classify SaaS intent: {e}")
            return await self._create_error_classification(intent_id, raw_input, str(e))
    
    async def _calculate_pattern_confidence(self, raw_input: str, pattern: SaaSIntentPattern) -> float:
        """Calculate confidence score for a pattern match"""
        try:
            confidence_factors = []
            
            # Regex pattern matching
            regex_matches = 0
            for regex_pattern in pattern.regex_patterns:
                if re.search(regex_pattern, raw_input):
                    regex_matches += 1
            
            if regex_matches > 0:
                regex_confidence = min(1.0, regex_matches / len(pattern.regex_patterns))
                confidence_factors.append(regex_confidence * 0.6)  # 60% weight for regex
            
            # Keyword matching
            keyword_matches = 0
            input_lower = raw_input.lower()
            for keyword in pattern.keywords:
                if keyword.lower() in input_lower:
                    keyword_matches += 1
            
            if keyword_matches > 0:
                keyword_confidence = min(1.0, keyword_matches / len(pattern.keywords))
                confidence_factors.append(keyword_confidence * 0.3)  # 30% weight for keywords
            
            # Pattern weight
            confidence_factors.append(pattern.confidence_weight * 0.1)  # 10% weight for pattern quality
            
            # Calculate final confidence
            if confidence_factors:
                final_confidence = sum(confidence_factors)
                return min(1.0, final_confidence)
            
            return 0.0
            
        except Exception as e:
            self.logger.error(f"❌ Error calculating pattern confidence: {e}")
            return 0.0
    
    async def _extract_parameters(self, raw_input: str, pattern: SaaSIntentPattern) -> ExtractedParameters:
        """Extract parameters from raw input using pattern extractors"""
        try:
            params = ExtractedParameters()
            
            # Extract using pattern-specific extractors
            for param_name, extractor_pattern in pattern.parameter_extractors.items():
                match = re.search(extractor_pattern, raw_input)
                if match:
                    if param_name == "time_period":
                        params.time_period = match.group(0).strip()
                    elif param_name == "metric_type":
                        params.metric_type = match.group(0).strip()
                    elif param_name == "account_name":
                        params.account_name = match.group(1).strip() if len(match.groups()) > 0 else match.group(0).strip()
                    elif param_name == "segment":
                        params.segment = match.group(0).strip()
                    elif param_name == "threshold":
                        try:
                            params.threshold = float(match.group(1) if len(match.groups()) > 0 else match.group(0))
                        except (ValueError, IndexError):
                            params.threshold = None
                    elif param_name == "team":
                        params.team = match.group(0).strip()
                    elif param_name == "region":
                        params.region = match.group(0).strip()
                    else:
                        params.custom_parameters[param_name] = match.group(0).strip()
            
            # Extract common parameters using general extractors
            await self._extract_common_parameters(raw_input, params)
            
            return params
            
        except Exception as e:
            self.logger.error(f"❌ Error extracting parameters: {e}")
            return ExtractedParameters()
    
    async def _extract_common_parameters(self, raw_input: str, params: ExtractedParameters):
        """Extract common parameters using general extractors"""
        try:
            # Extract currency amounts
            currency_match = re.search(r"(?i)\$(\d+(?:,\d{3})*(?:\.\d{2})?)", raw_input)
            if currency_match:
                params.currency = "USD"
                params.custom_parameters["amount"] = currency_match.group(1)
            
            # Extract percentages
            percent_match = re.search(r"(\d+(?:\.\d+)?)\s*(?:%|percent)", raw_input)
            if percent_match:
                params.custom_parameters["percentage"] = float(percent_match.group(1))
            
            # Extract product names
            product_match = re.search(r"(?i)\b(?:product|platform|solution)\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)", raw_input)
            if product_match:
                params.product = product_match.group(1)
            
        except Exception as e:
            self.logger.error(f"❌ Error extracting common parameters: {e}")
    
    async def _determine_compliance_flags(
        self, 
        category: SaaSIntentCategory, 
        params: ExtractedParameters, 
        context: Dict[str, Any]
    ) -> List[str]:
        """Determine compliance flags based on intent category and parameters"""
        flags = []
        
        try:
            # SOX compliance for financial data
            if category in [SaaSIntentCategory.ARR_ANALYSIS, SaaSIntentCategory.REVENUE_RECOGNITION, 
                          SaaSIntentCategory.COMP_PLAN, SaaSIntentCategory.FORECAST_ACCURACY]:
                flags.append("SOX")
            
            # GDPR compliance for customer data
            if category in [SaaSIntentCategory.CHURN_PREDICTION, SaaSIntentCategory.CUSTOMER_SUCCESS,
                          SaaSIntentCategory.QBR, SaaSIntentCategory.USAGE_ANALYTICS]:
                flags.append("GDPR")
            
            # Industry-specific compliance
            if context.get("industry_code") == "SAAS":
                flags.append("SAAS_BUSINESS_RULES")
            
            # Data residency flags
            if context.get("region") in ["EU", "EMEA"]:
                flags.append("EU_DATA_RESIDENCY")
            
            return list(set(flags))  # Remove duplicates
            
        except Exception as e:
            self.logger.error(f"❌ Error determining compliance flags: {e}")
            return []
    
    async def _get_recommended_capability(
        self, 
        category: SaaSIntentCategory, 
        automation_route: AutomationRoute,
        context: Dict[str, Any]
    ) -> Optional[str]:
        """Get recommended capability based on intent category and automation route"""
        try:
            capability_mapping = {
                (SaaSIntentCategory.ARR_ANALYSIS, AutomationRoute.RBA): "ARR Growth Analyzer",
                (SaaSIntentCategory.ARR_ANALYSIS, AutomationRoute.RBIA): "ARR Variance Detector",
                (SaaSIntentCategory.CHURN_PREDICTION, AutomationRoute.RBIA): "Churn Prediction Engine",
                (SaaSIntentCategory.CHURN_PREDICTION, AutomationRoute.RBA): "Customer Health Scorer",
                (SaaSIntentCategory.COMP_PLAN, AutomationRoute.RBA): "Compensation Plan Builder",
                (SaaSIntentCategory.QBR, AutomationRoute.AALA): "QBR Assistant",
                (SaaSIntentCategory.PIPELINE_HYGIENE, AutomationRoute.RBA): "Pipeline Hygiene Engine",
                (SaaSIntentCategory.FORECAST_ACCURACY, AutomationRoute.RBIA): "Forecast Accuracy Analyzer"
            }
            
            return capability_mapping.get((category, automation_route))
            
        except Exception as e:
            self.logger.error(f"❌ Error getting recommended capability: {e}")
            return None
    
    async def _create_fallback_classification(
        self, 
        intent_id: str, 
        raw_input: str, 
        context: Dict[str, Any]
    ) -> SaaSIntentClassification:
        """Create fallback classification for unmatched intents"""
        return SaaSIntentClassification(
            intent_id=intent_id,
            category=SaaSIntentCategory.PIPELINE_HYGIENE,  # Default fallback
            automation_route=AutomationRoute.RBA,
            confidence_score=0.1,
            extracted_parameters=ExtractedParameters(),
            matched_patterns=[],
            business_context={"fallback": True, "requires_manual_review": True},
            compliance_flags=[],
            recommended_capability=None
        )
    
    async def _create_error_classification(
        self, 
        intent_id: str, 
        raw_input: str, 
        error_message: str
    ) -> SaaSIntentClassification:
        """Create error classification for failed processing"""
        return SaaSIntentClassification(
            intent_id=intent_id,
            category=SaaSIntentCategory.PIPELINE_HYGIENE,  # Default
            automation_route=AutomationRoute.RBA,
            confidence_score=0.0,
            extracted_parameters=ExtractedParameters(),
            matched_patterns=[],
            business_context={"error": True, "error_message": error_message},
            compliance_flags=[],
            recommended_capability=None
        )
    
    def get_supported_categories(self) -> List[str]:
        """Get list of supported SaaS intent categories"""
        return [category.value for category in SaaSIntentCategory]
    
    def get_category_patterns(self, category: SaaSIntentCategory) -> List[SaaSIntentPattern]:
        """Get patterns for a specific category"""
        return [pattern for pattern in self.intent_patterns if pattern.category == category]
    
    async def validate_classification(self, classification: SaaSIntentClassification) -> Dict[str, Any]:
        """Validate classification results and provide feedback"""
        try:
            validation_result = {
                "is_valid": True,
                "confidence_assessment": "acceptable",
                "recommendations": [],
                "warnings": []
            }
            
            # Check confidence score
            if classification.confidence_score < 0.5:
                validation_result["confidence_assessment"] = "low"
                validation_result["recommendations"].append("Consider manual review")
            elif classification.confidence_score < 0.7:
                validation_result["confidence_assessment"] = "moderate"
            else:
                validation_result["confidence_assessment"] = "high"
            
            # Check for missing parameters
            if not classification.extracted_parameters.time_period and classification.category in [
                SaaSIntentCategory.ARR_ANALYSIS, SaaSIntentCategory.FORECAST_ACCURACY
            ]:
                validation_result["warnings"].append("Time period not specified - may default to current period")
            
            # Check compliance requirements
            if classification.compliance_flags and not classification.business_context.get("approval_required"):
                validation_result["warnings"].append("Compliance requirements detected - approval may be required")
            
            return validation_result
            
        except Exception as e:
            self.logger.error(f"❌ Error validating classification: {e}")
            return {"is_valid": False, "error": str(e)}

# Singleton instance
_saas_intent_taxonomy = None

def get_saas_intent_taxonomy() -> SaaSIntentTaxonomy:
    """Get singleton instance of SaaS Intent Taxonomy"""
    global _saas_intent_taxonomy
    if _saas_intent_taxonomy is None:
        _saas_intent_taxonomy = SaaSIntentTaxonomy()
    return _saas_intent_taxonomy
