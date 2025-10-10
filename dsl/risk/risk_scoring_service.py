#!/usr/bin/env python3
"""
Risk Scoring Service - Task 7.6-T20
===================================
Risk scoring service with standardized scoring algorithms
Linked to risk register schema for comprehensive risk management

Features:
- Dynamic risk scoring algorithms
- Multi-dimensional risk assessment
- Industry-specific risk models
- Real-time risk calculation
- Historical risk trend analysis
- Automated risk threshold alerts
"""

from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional, Union, Tuple, Callable
from enum import Enum
from datetime import datetime, timezone, timedelta
import math
import logging
import uuid
import json

logger = logging.getLogger(__name__)

# =====================================================
# TASK 7.6-T20: RISK SCORING SERVICE
# =====================================================

class RiskCategory(str, Enum):
    """Risk categories"""
    OPERATIONAL = "operational"
    COMPLIANCE = "compliance"
    SECURITY = "security"
    DATA = "data"
    SLA = "sla"
    FINANCIAL = "financial"
    REPUTATIONAL = "reputational"

class RiskLikelihood(str, Enum):
    """Risk likelihood levels"""
    VERY_LOW = "very_low"      # 0-5%
    LOW = "low"                # 5-25%
    MEDIUM = "medium"          # 25-50%
    HIGH = "high"              # 50-75%
    VERY_HIGH = "very_high"    # 75-100%

class RiskImpact(str, Enum):
    """Risk impact levels"""
    NEGLIGIBLE = "negligible"  # Minimal impact
    MINOR = "minor"           # Low impact
    MODERATE = "moderate"     # Medium impact
    MAJOR = "major"           # High impact
    CATASTROPHIC = "catastrophic"  # Severe impact

class RiskSeverity(str, Enum):
    """Overall risk severity"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

class IndustryCode(str, Enum):
    """Industry codes for risk scoring"""
    SAAS = "SaaS"
    BANKING = "Banking"
    INSURANCE = "Insurance"
    HEALTHCARE = "Healthcare"
    ECOMMERCE = "E-commerce"
    FINTECH = "FinTech"

@dataclass
class RiskFactor:
    """Individual risk factor"""
    factor_id: str
    factor_name: str
    category: RiskCategory
    weight: float              # 0.0 to 1.0
    current_value: float       # Actual measured value
    threshold_low: float       # Low risk threshold
    threshold_medium: float    # Medium risk threshold
    threshold_high: float      # High risk threshold
    threshold_critical: float  # Critical risk threshold
    measurement_unit: str
    description: str
    
    def calculate_factor_score(self) -> Tuple[float, RiskSeverity]:
        """Calculate risk score for this factor"""
        if self.current_value <= self.threshold_low:
            return 0.1, RiskSeverity.LOW
        elif self.current_value <= self.threshold_medium:
            return 0.3, RiskSeverity.MEDIUM
        elif self.current_value <= self.threshold_high:
            return 0.7, RiskSeverity.HIGH
        else:
            return 1.0, RiskSeverity.CRITICAL

@dataclass
class RiskScoreComponents:
    """Components of a risk score"""
    base_score: float
    likelihood_score: float
    impact_score: float
    temporal_adjustment: float
    industry_adjustment: float
    mitigation_adjustment: float
    final_score: float
    confidence_level: float

@dataclass
class RiskAssessment:
    """Complete risk assessment result"""
    risk_id: str
    overall_score: float       # 0.0 to 1.0
    severity: RiskSeverity
    likelihood: RiskLikelihood
    impact: RiskImpact
    score_components: RiskScoreComponents
    contributing_factors: List[RiskFactor]
    recommendations: List[str]
    assessment_timestamp: str
    valid_until: str
    assessor: str

class RiskScoringService:
    """
    Task 7.6-T20: Risk Scoring Service Implementation
    Comprehensive risk scoring with industry-specific models
    """
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.scoring_models = self._initialize_scoring_models()
        self.industry_adjustments = self._initialize_industry_adjustments()
        self.risk_thresholds = self._initialize_risk_thresholds()
        
        # Scoring history for trend analysis
        self.scoring_history: Dict[str, List[RiskAssessment]] = {}
    
    def _initialize_scoring_models(self) -> Dict[str, Dict[str, Any]]:
        """Initialize risk scoring models"""
        
        models = {}
        
        # BASE SCORING MODEL
        models["base"] = {
            "likelihood_weights": {
                RiskLikelihood.VERY_LOW: 0.05,
                RiskLikelihood.LOW: 0.2,
                RiskLikelihood.MEDIUM: 0.45,
                RiskLikelihood.HIGH: 0.7,
                RiskLikelihood.VERY_HIGH: 0.9
            },
            "impact_weights": {
                RiskImpact.NEGLIGIBLE: 0.1,
                RiskImpact.MINOR: 0.3,
                RiskImpact.MODERATE: 0.5,
                RiskImpact.MAJOR: 0.7,
                RiskImpact.CATASTROPHIC: 0.9
            },
            "category_weights": {
                RiskCategory.OPERATIONAL: 0.8,
                RiskCategory.COMPLIANCE: 1.0,
                RiskCategory.SECURITY: 0.9,
                RiskCategory.DATA: 0.8,
                RiskCategory.SLA: 0.7,
                RiskCategory.FINANCIAL: 0.9,
                RiskCategory.REPUTATIONAL: 0.6
            }
        }
        
        # OPERATIONAL RISK MODEL
        models["operational"] = {
            "factors": [
                {
                    "factor_id": "system_availability",
                    "weight": 0.3,
                    "thresholds": {"low": 99.9, "medium": 99.5, "high": 99.0, "critical": 95.0}
                },
                {
                    "factor_id": "error_rate",
                    "weight": 0.25,
                    "thresholds": {"low": 0.1, "medium": 0.5, "high": 1.0, "critical": 5.0}
                },
                {
                    "factor_id": "response_time",
                    "weight": 0.2,
                    "thresholds": {"low": 200, "medium": 500, "high": 1000, "critical": 5000}
                },
                {
                    "factor_id": "resource_utilization",
                    "weight": 0.15,
                    "thresholds": {"low": 70, "medium": 80, "high": 90, "critical": 95}
                },
                {
                    "factor_id": "incident_frequency",
                    "weight": 0.1,
                    "thresholds": {"low": 1, "medium": 3, "high": 5, "critical": 10}
                }
            ]
        }
        
        # COMPLIANCE RISK MODEL
        models["compliance"] = {
            "factors": [
                {
                    "factor_id": "policy_violations",
                    "weight": 0.4,
                    "thresholds": {"low": 0, "medium": 1, "high": 3, "critical": 5}
                },
                {
                    "factor_id": "audit_findings",
                    "weight": 0.3,
                    "thresholds": {"low": 0, "medium": 1, "high": 2, "critical": 5}
                },
                {
                    "factor_id": "regulatory_changes",
                    "weight": 0.2,
                    "thresholds": {"low": 0, "medium": 1, "high": 2, "critical": 3}
                },
                {
                    "factor_id": "certification_status",
                    "weight": 0.1,
                    "thresholds": {"low": 100, "medium": 90, "high": 80, "critical": 70}
                }
            ]
        }
        
        # SECURITY RISK MODEL
        models["security"] = {
            "factors": [
                {
                    "factor_id": "vulnerability_count",
                    "weight": 0.3,
                    "thresholds": {"low": 0, "medium": 2, "high": 5, "critical": 10}
                },
                {
                    "factor_id": "security_incidents",
                    "weight": 0.25,
                    "thresholds": {"low": 0, "medium": 1, "high": 2, "critical": 5}
                },
                {
                    "factor_id": "patch_compliance",
                    "weight": 0.2,
                    "thresholds": {"low": 95, "medium": 90, "high": 85, "critical": 80}
                },
                {
                    "factor_id": "access_violations",
                    "weight": 0.15,
                    "thresholds": {"low": 0, "medium": 1, "high": 3, "critical": 5}
                },
                {
                    "factor_id": "threat_level",
                    "weight": 0.1,
                    "thresholds": {"low": 1, "medium": 2, "high": 3, "critical": 4}
                }
            ]
        }
        
        return models
    
    def _initialize_industry_adjustments(self) -> Dict[IndustryCode, Dict[str, float]]:
        """Initialize industry-specific risk adjustments"""
        
        adjustments = {}
        
        # Banking industry - higher risk sensitivity
        adjustments[IndustryCode.BANKING] = {
            "compliance_multiplier": 1.5,
            "security_multiplier": 1.3,
            "operational_multiplier": 1.2,
            "base_threshold_adjustment": -0.1  # Lower thresholds (more sensitive)
        }
        
        # Insurance industry - moderate risk sensitivity
        adjustments[IndustryCode.INSURANCE] = {
            "compliance_multiplier": 1.3,
            "security_multiplier": 1.2,
            "operational_multiplier": 1.1,
            "base_threshold_adjustment": -0.05
        }
        
        # Healthcare - highest risk sensitivity
        adjustments[IndustryCode.HEALTHCARE] = {
            "compliance_multiplier": 1.6,
            "security_multiplier": 1.4,
            "operational_multiplier": 1.3,
            "base_threshold_adjustment": -0.15
        }
        
        # SaaS - moderate risk tolerance
        adjustments[IndustryCode.SAAS] = {
            "compliance_multiplier": 1.0,
            "security_multiplier": 1.1,
            "operational_multiplier": 1.0,
            "base_threshold_adjustment": 0.0
        }
        
        # E-commerce - balanced approach
        adjustments[IndustryCode.ECOMMERCE] = {
            "compliance_multiplier": 1.1,
            "security_multiplier": 1.2,
            "operational_multiplier": 1.0,
            "base_threshold_adjustment": 0.0
        }
        
        # FinTech - high risk sensitivity
        adjustments[IndustryCode.FINTECH] = {
            "compliance_multiplier": 1.4,
            "security_multiplier": 1.3,
            "operational_multiplier": 1.2,
            "base_threshold_adjustment": -0.1
        }
        
        return adjustments
    
    def _initialize_risk_thresholds(self) -> Dict[str, Dict[str, float]]:
        """Initialize risk scoring thresholds"""
        
        return {
            "severity_thresholds": {
                "low": 0.25,
                "medium": 0.5,
                "high": 0.75,
                "critical": 1.0
            },
            "confidence_thresholds": {
                "low": 0.6,
                "medium": 0.75,
                "high": 0.9
            },
            "alert_thresholds": {
                "warning": 0.6,
                "critical": 0.8
            }
        }
    
    def calculate_risk_score(
        self,
        risk_id: str,
        risk_category: RiskCategory,
        likelihood: RiskLikelihood,
        impact: RiskImpact,
        risk_factors: List[Dict[str, Any]],
        context: Optional[Dict[str, Any]] = None
    ) -> RiskAssessment:
        """
        Calculate comprehensive risk score
        """
        try:
            context = context or {}
            assessment_time = datetime.now(timezone.utc)
            
            # Get scoring model
            model = self.scoring_models.get(risk_category.value, self.scoring_models["base"])
            
            # Calculate base score components
            likelihood_score = model["likelihood_weights"][likelihood]
            impact_score = model["impact_weights"][impact]
            category_weight = model["category_weights"].get(risk_category, 0.8)
            
            # Calculate base risk score
            base_score = (likelihood_score * impact_score * category_weight)
            
            # Process risk factors
            processed_factors = []
            factor_adjustment = 0.0
            
            for factor_data in risk_factors:
                factor = self._create_risk_factor(factor_data, risk_category)
                processed_factors.append(factor)
                
                factor_score, _ = factor.calculate_factor_score()
                factor_adjustment += (factor_score * factor.weight)
            
            # Normalize factor adjustment
            if processed_factors:
                factor_adjustment = factor_adjustment / len(processed_factors)
            
            # Apply temporal adjustment (recent events have higher impact)
            temporal_adjustment = self._calculate_temporal_adjustment(context)
            
            # Apply industry-specific adjustments
            industry_code = context.get("industry_code")
            industry_adjustment = self._calculate_industry_adjustment(
                base_score, risk_category, industry_code
            )
            
            # Apply mitigation adjustments
            mitigation_adjustment = self._calculate_mitigation_adjustment(context)
            
            # Calculate final score
            final_score = min(1.0, max(0.0, 
                base_score + 
                (factor_adjustment * 0.3) + 
                (temporal_adjustment * 0.1) + 
                (industry_adjustment * 0.2) + 
                (mitigation_adjustment * -0.2)  # Negative because mitigation reduces risk
            ))
            
            # Determine final severity
            final_severity = self._determine_severity(final_score)
            
            # Calculate confidence level
            confidence_level = self._calculate_confidence_level(
                processed_factors, context
            )
            
            # Generate recommendations
            recommendations = self._generate_recommendations(
                final_score, final_severity, risk_category, processed_factors
            )
            
            # Create score components
            score_components = RiskScoreComponents(
                base_score=base_score,
                likelihood_score=likelihood_score,
                impact_score=impact_score,
                temporal_adjustment=temporal_adjustment,
                industry_adjustment=industry_adjustment,
                mitigation_adjustment=mitigation_adjustment,
                final_score=final_score,
                confidence_level=confidence_level
            )
            
            # Create assessment
            assessment = RiskAssessment(
                risk_id=risk_id,
                overall_score=final_score,
                severity=final_severity,
                likelihood=likelihood,
                impact=impact,
                score_components=score_components,
                contributing_factors=processed_factors,
                recommendations=recommendations,
                assessment_timestamp=assessment_time.isoformat(),
                valid_until=(assessment_time + timedelta(hours=24)).isoformat(),
                assessor="risk_scoring_service"
            )
            
            # Store in history
            if risk_id not in self.scoring_history:
                self.scoring_history[risk_id] = []
            self.scoring_history[risk_id].append(assessment)
            
            # Keep only last 100 assessments per risk
            if len(self.scoring_history[risk_id]) > 100:
                self.scoring_history[risk_id] = self.scoring_history[risk_id][-100:]
            
            return assessment
            
        except Exception as e:
            self.logger.error(f"Error calculating risk score for {risk_id}: {e}")
            
            # Return default high-risk assessment on error
            return RiskAssessment(
                risk_id=risk_id,
                overall_score=0.8,
                severity=RiskSeverity.HIGH,
                likelihood=likelihood,
                impact=impact,
                score_components=RiskScoreComponents(
                    base_score=0.8, likelihood_score=0.0, impact_score=0.0,
                    temporal_adjustment=0.0, industry_adjustment=0.0,
                    mitigation_adjustment=0.0, final_score=0.8, confidence_level=0.3
                ),
                contributing_factors=[],
                recommendations=["Error in risk calculation - manual review required"],
                assessment_timestamp=datetime.now(timezone.utc).isoformat(),
                valid_until=(datetime.now(timezone.utc) + timedelta(hours=1)).isoformat(),
                assessor="risk_scoring_service_error"
            )
    
    def _create_risk_factor(self, factor_data: Dict[str, Any], category: RiskCategory) -> RiskFactor:
        """Create risk factor from data"""
        
        return RiskFactor(
            factor_id=factor_data.get("factor_id", str(uuid.uuid4())),
            factor_name=factor_data.get("factor_name", "Unknown Factor"),
            category=category,
            weight=factor_data.get("weight", 0.5),
            current_value=factor_data.get("current_value", 0.0),
            threshold_low=factor_data.get("threshold_low", 0.25),
            threshold_medium=factor_data.get("threshold_medium", 0.5),
            threshold_high=factor_data.get("threshold_high", 0.75),
            threshold_critical=factor_data.get("threshold_critical", 1.0),
            measurement_unit=factor_data.get("measurement_unit", "units"),
            description=factor_data.get("description", "Risk factor")
        )
    
    def _calculate_temporal_adjustment(self, context: Dict[str, Any]) -> float:
        """Calculate temporal adjustment based on recent events"""
        
        recent_incidents = context.get("recent_incidents", 0)
        days_since_last_incident = context.get("days_since_last_incident", 30)
        
        # Recent incidents increase risk
        incident_adjustment = min(0.2, recent_incidents * 0.05)
        
        # Time decay - older incidents have less impact
        time_decay = max(0.0, 1.0 - (days_since_last_incident / 30.0))
        
        return incident_adjustment * time_decay
    
    def _calculate_industry_adjustment(
        self,
        base_score: float,
        category: RiskCategory,
        industry_code: Optional[str]
    ) -> float:
        """Calculate industry-specific risk adjustment"""
        
        if not industry_code:
            return 0.0
        
        try:
            industry_enum = IndustryCode(industry_code)
            adjustments = self.industry_adjustments.get(industry_enum, {})
            
            # Get category-specific multiplier
            multiplier_key = f"{category.value}_multiplier"
            multiplier = adjustments.get(multiplier_key, 1.0)
            
            # Apply base threshold adjustment
            threshold_adjustment = adjustments.get("base_threshold_adjustment", 0.0)
            
            # Calculate adjustment
            adjustment = (base_score * (multiplier - 1.0)) + threshold_adjustment
            
            return max(-0.3, min(0.3, adjustment))  # Cap adjustment at ±0.3
            
        except ValueError:
            return 0.0
    
    def _calculate_mitigation_adjustment(self, context: Dict[str, Any]) -> float:
        """Calculate mitigation effectiveness adjustment"""
        
        mitigation_measures = context.get("mitigation_measures", [])
        if not mitigation_measures:
            return 0.0
        
        total_effectiveness = 0.0
        
        for measure in mitigation_measures:
            effectiveness = measure.get("effectiveness", 0.0)  # 0.0 to 1.0
            implementation_status = measure.get("implementation_status", 0.0)  # 0.0 to 1.0
            
            # Mitigation effectiveness is product of planned effectiveness and implementation
            actual_effectiveness = effectiveness * implementation_status
            total_effectiveness += actual_effectiveness
        
        # Average effectiveness across all measures
        if mitigation_measures:
            avg_effectiveness = total_effectiveness / len(mitigation_measures)
            return min(0.5, avg_effectiveness)  # Cap at 50% risk reduction
        
        return 0.0
    
    def _determine_severity(self, score: float) -> RiskSeverity:
        """Determine risk severity from score"""
        
        thresholds = self.risk_thresholds["severity_thresholds"]
        
        if score >= thresholds["critical"]:
            return RiskSeverity.CRITICAL
        elif score >= thresholds["high"]:
            return RiskSeverity.HIGH
        elif score >= thresholds["medium"]:
            return RiskSeverity.MEDIUM
        else:
            return RiskSeverity.LOW
    
    def _calculate_confidence_level(
        self,
        factors: List[RiskFactor],
        context: Dict[str, Any]
    ) -> float:
        """Calculate confidence level of risk assessment"""
        
        confidence_factors = []
        
        # Data quality factor
        data_quality = context.get("data_quality", 0.8)
        confidence_factors.append(data_quality)
        
        # Number of factors factor (more factors = higher confidence)
        factor_count_confidence = min(1.0, len(factors) / 5.0)
        confidence_factors.append(factor_count_confidence)
        
        # Historical data availability
        historical_data_months = context.get("historical_data_months", 3)
        historical_confidence = min(1.0, historical_data_months / 12.0)
        confidence_factors.append(historical_confidence)
        
        # Expert validation
        expert_validated = context.get("expert_validated", False)
        if expert_validated:
            confidence_factors.append(0.9)
        else:
            confidence_factors.append(0.6)
        
        # Calculate average confidence
        return sum(confidence_factors) / len(confidence_factors)
    
    def _generate_recommendations(
        self,
        score: float,
        severity: RiskSeverity,
        category: RiskCategory,
        factors: List[RiskFactor]
    ) -> List[str]:
        """Generate risk mitigation recommendations"""
        
        recommendations = []
        
        # Severity-based recommendations
        if severity == RiskSeverity.CRITICAL:
            recommendations.append("Immediate action required - escalate to executive team")
            recommendations.append("Implement emergency response procedures")
        elif severity == RiskSeverity.HIGH:
            recommendations.append("High priority - assign dedicated resources")
            recommendations.append("Develop comprehensive mitigation plan within 48 hours")
        elif severity == RiskSeverity.MEDIUM:
            recommendations.append("Schedule mitigation planning within one week")
            recommendations.append("Monitor closely for escalation")
        else:
            recommendations.append("Continue routine monitoring")
            recommendations.append("Review mitigation measures quarterly")
        
        # Category-specific recommendations
        if category == RiskCategory.OPERATIONAL:
            recommendations.append("Review system monitoring and alerting")
            recommendations.append("Consider redundancy and failover mechanisms")
        elif category == RiskCategory.COMPLIANCE:
            recommendations.append("Engage compliance team for detailed review")
            recommendations.append("Update policies and procedures as needed")
        elif category == RiskCategory.SECURITY:
            recommendations.append("Conduct security assessment")
            recommendations.append("Review access controls and permissions")
        elif category == RiskCategory.DATA:
            recommendations.append("Implement data quality monitoring")
            recommendations.append("Review data governance policies")
        
        # Factor-specific recommendations
        high_risk_factors = [f for f in factors if f.calculate_factor_score()[1] in [RiskSeverity.HIGH, RiskSeverity.CRITICAL]]
        
        for factor in high_risk_factors:
            recommendations.append(f"Address high-risk factor: {factor.factor_name}")
        
        return recommendations
    
    def get_risk_trend(self, risk_id: str, days: int = 30) -> Dict[str, Any]:
        """Get risk score trend over time"""
        
        if risk_id not in self.scoring_history:
            return {
                "risk_id": risk_id,
                "trend_available": False,
                "message": "No historical data available"
            }
        
        history = self.scoring_history[risk_id]
        cutoff_date = datetime.now(timezone.utc) - timedelta(days=days)
        
        # Filter recent assessments
        recent_assessments = [
            a for a in history
            if datetime.fromisoformat(a.assessment_timestamp.replace('Z', '+00:00')) >= cutoff_date
        ]
        
        if len(recent_assessments) < 2:
            return {
                "risk_id": risk_id,
                "trend_available": False,
                "message": "Insufficient data for trend analysis"
            }
        
        # Calculate trend
        scores = [a.overall_score for a in recent_assessments]
        timestamps = [datetime.fromisoformat(a.assessment_timestamp.replace('Z', '+00:00')) for a in recent_assessments]
        
        # Simple linear trend calculation
        n = len(scores)
        sum_x = sum(range(n))
        sum_y = sum(scores)
        sum_xy = sum(i * score for i, score in enumerate(scores))
        sum_x2 = sum(i * i for i in range(n))
        
        slope = (n * sum_xy - sum_x * sum_y) / (n * sum_x2 - sum_x * sum_x)
        
        # Determine trend direction
        if slope > 0.01:
            trend_direction = "increasing"
        elif slope < -0.01:
            trend_direction = "decreasing"
        else:
            trend_direction = "stable"
        
        return {
            "risk_id": risk_id,
            "trend_available": True,
            "trend_direction": trend_direction,
            "trend_slope": slope,
            "current_score": scores[-1],
            "previous_score": scores[0],
            "score_change": scores[-1] - scores[0],
            "assessment_count": len(recent_assessments),
            "period_days": days,
            "last_assessment": recent_assessments[-1].assessment_timestamp
        }
    
    def get_risk_distribution(self, tenant_id: Optional[int] = None) -> Dict[str, Any]:
        """Get risk distribution across categories and severities"""
        
        # In practice, this would query the database
        # For now, return mock distribution based on scoring history
        
        distribution = {
            "by_severity": {
                "low": 0,
                "medium": 0,
                "high": 0,
                "critical": 0
            },
            "by_category": {
                category.value: 0 for category in RiskCategory
            },
            "total_risks": 0,
            "average_score": 0.0,
            "tenant_id": tenant_id
        }
        
        all_assessments = []
        for risk_assessments in self.scoring_history.values():
            if risk_assessments:
                all_assessments.append(risk_assessments[-1])  # Latest assessment
        
        if not all_assessments:
            return distribution
        
        # Calculate distribution
        for assessment in all_assessments:
            distribution["by_severity"][assessment.severity.value] += 1
            distribution["total_risks"] += 1
        
        # Calculate average score
        if all_assessments:
            distribution["average_score"] = sum(a.overall_score for a in all_assessments) / len(all_assessments)
        
        return distribution
    
    def check_risk_thresholds(self, assessment: RiskAssessment) -> Dict[str, Any]:
        """Check if risk assessment triggers any thresholds"""
        
        alerts = []
        
        # Check alert thresholds
        if assessment.overall_score >= self.risk_thresholds["alert_thresholds"]["critical"]:
            alerts.append({
                "level": "critical",
                "message": f"Risk {assessment.risk_id} exceeds critical threshold",
                "score": assessment.overall_score,
                "threshold": self.risk_thresholds["alert_thresholds"]["critical"]
            })
        elif assessment.overall_score >= self.risk_thresholds["alert_thresholds"]["warning"]:
            alerts.append({
                "level": "warning",
                "message": f"Risk {assessment.risk_id} exceeds warning threshold",
                "score": assessment.overall_score,
                "threshold": self.risk_thresholds["alert_thresholds"]["warning"]
            })
        
        # Check confidence thresholds
        if assessment.score_components.confidence_level < self.risk_thresholds["confidence_thresholds"]["low"]:
            alerts.append({
                "level": "info",
                "message": f"Low confidence in risk assessment for {assessment.risk_id}",
                "confidence": assessment.score_components.confidence_level,
                "threshold": self.risk_thresholds["confidence_thresholds"]["low"]
            })
        
        return {
            "risk_id": assessment.risk_id,
            "alerts_triggered": len(alerts),
            "alerts": alerts,
            "requires_immediate_attention": any(alert["level"] == "critical" for alert in alerts)
        }

# Initialize global risk scoring service
risk_scoring_service = RiskScoringService()

def get_risk_scoring_service() -> RiskScoringService:
    """Get risk scoring service instance"""
    return risk_scoring_service
