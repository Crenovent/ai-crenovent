"""
Comprehensive Sandbagging Analyzer
Implements advanced multi-dimensional sandbagging detection using 100+ parameters
"""

import logging
import asyncio
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
from dataclasses import dataclass

from dsl.parameters.dynamic_parameter_discovery import parameter_discovery, ParameterDefinition, ParameterCategory
from dsl.data_sources.data_source_manager import data_source_manager

logger = logging.getLogger(__name__)

@dataclass
class SandbaggingAssessment:
    """Comprehensive sandbagging assessment result"""
    opportunity_id: str
    overall_score: float
    risk_level: str  # LOW, MEDIUM, HIGH, CRITICAL
    confidence: float
    
    # Category-specific scores
    deal_level_score: float
    rep_behavior_score: float
    temporal_score: float
    organizational_score: float
    market_score: float
    
    # Parameter details
    active_parameters: List[str]
    parameter_values: Dict[str, Any]
    contributing_factors: List[str]
    
    # Recommendations
    recommendations: List[str]
    next_actions: List[str]
    
    # Metadata
    analysis_timestamp: str
    confidence_factors: List[str]

class ComprehensiveSandbaggingAnalyzer:
    """
    Advanced sandbagging analyzer using 100+ dynamic parameters
    Provides multi-dimensional analysis with contextual intelligence
    """
    
    def __init__(self):
        self.parameter_discovery = parameter_discovery
        self.data_source_manager = data_source_manager
        
        # Scoring weights for different parameter categories
        self.category_weights = {
            ParameterCategory.DEAL_LEVEL: 0.30,
            ParameterCategory.REP_BEHAVIOR: 0.25,
            ParameterCategory.TEMPORAL_PATTERNS: 0.20,
            ParameterCategory.ORGANIZATIONAL: 0.15,
            ParameterCategory.MARKET_CUSTOMER: 0.10
        }
        
        # Risk level thresholds
        self.risk_thresholds = {
            'LOW': 0.3,
            'MEDIUM': 0.5,
            'HIGH': 0.7,
            'CRITICAL': 0.85
        }
    
    async def analyze_comprehensive_sandbagging(
        self, 
        opportunities: List[Dict[str, Any]], 
        config: Dict[str, Any]
    ) -> List[SandbaggingAssessment]:
        """
        Perform comprehensive sandbagging analysis on opportunities
        """
        logger.info(f"🔍 Starting comprehensive sandbagging analysis on {len(opportunities)} opportunities")
        
        assessments = []
        expertise_level = config.get('expertise_level', 'expert')
        
        for opportunity in opportunities:
            try:
                assessment = await self._analyze_single_opportunity(opportunity, config, expertise_level)
                assessments.append(assessment)
                
            except Exception as e:
                logger.error(f"Failed to analyze opportunity {opportunity.get('Id', 'unknown')}: {e}")
                # Create fallback assessment
                fallback_assessment = self._create_fallback_assessment(opportunity)
                assessments.append(fallback_assessment)
        
        # Sort by overall score (highest risk first)
        assessments.sort(key=lambda x: x.overall_score, reverse=True)
        
        logger.info(f"✅ Completed comprehensive analysis. Found {sum(1 for a in assessments if a.risk_level in ['HIGH', 'CRITICAL'])} high-risk deals")
        
        return assessments
    
    async def _analyze_single_opportunity(
        self, 
        opportunity: Dict[str, Any], 
        config: Dict[str, Any], 
        expertise_level: str
    ) -> SandbaggingAssessment:
        """Analyze a single opportunity comprehensively"""
        
        opp_id = opportunity.get('Id', f"opp_{hash(str(opportunity))}")
        
        # Get contextual data for this deal
        contextual_data = await self.data_source_manager.get_contextual_data_for_deal(opportunity)
        
        # Get active parameters for this deal context
        active_parameters = await self._get_contextual_parameters(opportunity, contextual_data, expertise_level)
        
        # Calculate category-specific scores
        category_scores = await self._calculate_category_scores(opportunity, contextual_data, active_parameters, config)
        
        # Calculate overall composite score
        overall_score = self._calculate_composite_score(category_scores)
        
        # Determine risk level and confidence
        risk_level = self._determine_risk_level(overall_score)
        confidence = self._calculate_confidence(active_parameters, contextual_data)
        
        # Generate insights and recommendations
        contributing_factors = self._identify_contributing_factors(category_scores, active_parameters)
        recommendations = self._generate_recommendations(risk_level, contributing_factors, opportunity)
        next_actions = self._generate_next_actions(risk_level, opportunity)
        confidence_factors = self._identify_confidence_factors(contextual_data, active_parameters)
        
        # Extract parameter values for transparency
        parameter_values = self._extract_parameter_values(opportunity, contextual_data, active_parameters)
        
        return SandbaggingAssessment(
            opportunity_id=opp_id,
            overall_score=overall_score,
            risk_level=risk_level,
            confidence=confidence,
            deal_level_score=category_scores.get(ParameterCategory.DEAL_LEVEL, 0.0),
            rep_behavior_score=category_scores.get(ParameterCategory.REP_BEHAVIOR, 0.0),
            temporal_score=category_scores.get(ParameterCategory.TEMPORAL_PATTERNS, 0.0),
            organizational_score=category_scores.get(ParameterCategory.ORGANIZATIONAL, 0.0),
            market_score=category_scores.get(ParameterCategory.MARKET_CUSTOMER, 0.0),
            active_parameters=[p.key for p in active_parameters],
            parameter_values=parameter_values,
            contributing_factors=contributing_factors,
            recommendations=recommendations,
            next_actions=next_actions,
            analysis_timestamp=datetime.now().isoformat(),
            confidence_factors=confidence_factors
        )
    
    async def _get_contextual_parameters(
        self, 
        opportunity: Dict[str, Any], 
        contextual_data: Dict[str, Any], 
        expertise_level: str
    ) -> List[ParameterDefinition]:
        """Get contextually relevant parameters for this opportunity"""
        
        # Get base parameters for expertise level
        all_params = self.parameter_discovery._generate_sandbagging_parameters(expertise_level)
        
        # Filter parameters based on data availability
        available_sources = self.data_source_manager.get_available_parameter_sources()
        
        active_params = []
        for param in all_params:
            # Check if we have required data for this parameter
            if self._can_calculate_parameter(param, opportunity, contextual_data, available_sources):
                active_params.append(param)
        
        # Add contextual parameters based on deal characteristics
        contextual_params = self._get_deal_specific_parameters(opportunity, all_params)
        
        # Combine and deduplicate
        all_active_params = active_params + contextual_params
        unique_params = {p.key: p for p in all_active_params}.values()
        
        logger.debug(f"📊 Activated {len(unique_params)} parameters for opportunity {opportunity.get('Id', 'unknown')}")
        
        return list(unique_params)
    
    def _can_calculate_parameter(
        self, 
        param: ParameterDefinition, 
        opportunity: Dict[str, Any], 
        contextual_data: Dict[str, Any], 
        available_sources: List[str]
    ) -> bool:
        """Check if we can calculate this parameter with available data"""
        
        # Check data source requirements
        if param.data_source and param.data_source not in available_sources:
            return False
        
        # Check historical data requirements
        if param.requires_historical_data and not contextual_data.get('rep_history'):
            return False
        
        # Check external data requirements
        if param.requires_external_data and not contextual_data.get('market_data'):
            return False
        
        # Check field dependencies
        if param.depends_on_fields:
            for field in param.depends_on_fields:
                if field not in opportunity:
                    return False
        
        return True
    
    def _get_deal_specific_parameters(
        self, 
        opportunity: Dict[str, Any], 
        all_params: List[ParameterDefinition]
    ) -> List[ParameterDefinition]:
        """Get additional parameters based on deal characteristics"""
        
        contextual_params = []
        
        # High-value deal parameters
        amount = opportunity.get('Amount', 0)
        if amount > 500000:
            high_value_params = [p for p in all_params if 'high_value' in p.key or 'large_deal' in p.key]
            contextual_params.extend(high_value_params)
        
        # Advanced stage parameters
        stage = opportunity.get('StageName', '')
        advanced_stages = ['Proposal/Price Quote', 'Negotiation/Review', 'Id. Decision Makers']
        if stage in advanced_stages:
            stage_params = [p for p in all_params if 'stage' in p.key or 'advanced' in p.key]
            contextual_params.extend(stage_params)
        
        # Competitive deal parameters
        if opportunity.get('HasCompetition', False):
            competitive_params = [p for p in all_params if 'competitor' in p.key or 'competitive' in p.key]
            contextual_params.extend(competitive_params)
        
        return contextual_params
    
    async def _calculate_category_scores(
        self, 
        opportunity: Dict[str, Any], 
        contextual_data: Dict[str, Any], 
        active_parameters: List[ParameterDefinition], 
        config: Dict[str, Any]
    ) -> Dict[ParameterCategory, float]:
        """Calculate scores for each parameter category"""
        
        category_scores = {}
        
        # Group parameters by category
        params_by_category = {}
        for param in active_parameters:
            if param.category not in params_by_category:
                params_by_category[param.category] = []
            params_by_category[param.category].append(param)
        
        # Calculate score for each category
        for category, params in params_by_category.items():
            if category in [
                ParameterCategory.DEAL_LEVEL, 
                ParameterCategory.REP_BEHAVIOR, 
                ParameterCategory.TEMPORAL_PATTERNS,
                ParameterCategory.ORGANIZATIONAL, 
                ParameterCategory.MARKET_CUSTOMER
            ]:
                score = await self._calculate_category_specific_score(
                    category, params, opportunity, contextual_data, config
                )
                category_scores[category] = score
        
        return category_scores
    
    async def _calculate_category_specific_score(
        self, 
        category: ParameterCategory, 
        parameters: List[ParameterDefinition], 
        opportunity: Dict[str, Any], 
        contextual_data: Dict[str, Any], 
        config: Dict[str, Any]
    ) -> float:
        """Calculate score for a specific parameter category"""
        
        if not parameters:
            return 0.0
        
        category_score = 0.0
        total_weight = 0.0
        
        for param in parameters:
            try:
                # Get parameter value from config or default
                param_value = config.get(param.key, param.default_value)
                
                # Calculate parameter contribution to sandbagging score
                param_contribution = self._calculate_parameter_contribution(
                    param, param_value, opportunity, contextual_data
                )
                
                # Weight by parameter impact level
                weight = self._get_parameter_weight(param)
                category_score += param_contribution * weight
                total_weight += weight
                
            except Exception as e:
                logger.warning(f"Failed to calculate parameter {param.key}: {e}")
                continue
        
        # Normalize by total weight
        if total_weight > 0:
            category_score = category_score / total_weight
        
        # Ensure score is between 0 and 1
        return max(0.0, min(1.0, category_score))
    
    def _calculate_parameter_contribution(
        self, 
        param: ParameterDefinition, 
        param_value: Any, 
        opportunity: Dict[str, Any], 
        contextual_data: Dict[str, Any]
    ) -> float:
        """Calculate how much this parameter contributes to sandbagging score"""
        
        contribution = 0.0
        
        try:
            if param.category == ParameterCategory.DEAL_LEVEL:
                contribution = self._calculate_deal_level_contribution(param, param_value, opportunity, contextual_data)
            elif param.category == ParameterCategory.REP_BEHAVIOR:
                contribution = self._calculate_rep_behavior_contribution(param, param_value, opportunity, contextual_data)
            elif param.category == ParameterCategory.TEMPORAL_PATTERNS:
                contribution = self._calculate_temporal_contribution(param, param_value, opportunity, contextual_data)
            elif param.category == ParameterCategory.ORGANIZATIONAL:
                contribution = self._calculate_organizational_contribution(param, param_value, opportunity, contextual_data)
            elif param.category == ParameterCategory.MARKET_CUSTOMER:
                contribution = self._calculate_market_contribution(param, param_value, opportunity, contextual_data)
        
        except Exception as e:
            logger.warning(f"Error calculating contribution for {param.key}: {e}")
            contribution = 0.0
        
        return max(0.0, min(1.0, contribution))
    
    def _calculate_deal_level_contribution(
        self, 
        param: ParameterDefinition, 
        param_value: Any, 
        opportunity: Dict[str, Any], 
        contextual_data: Dict[str, Any]
    ) -> float:
        """Calculate deal-level parameter contribution"""
        
        if param.key == 'sandbagging_threshold':
            probability = opportunity.get('Probability', 50)
            return 1.0 if probability < param_value else 0.0
            
        elif param.key == 'high_value_threshold':
            amount = opportunity.get('Amount', 0)
            return 0.8 if amount > param_value else 0.0
            
        elif param.key == 'advanced_stage_multiplier':
            stage = opportunity.get('StageName', '')
            advanced_stages = ['Proposal/Price Quote', 'Negotiation/Review', 'Id. Decision Makers']
            if stage in advanced_stages:
                probability = opportunity.get('Probability', 50)
                return (100 - probability) / 100 * param_value
            return 0.0
            
        elif param.key == 'activity_recency_threshold_days':
            calculated_metrics = contextual_data.get('calculated_metrics', {})
            days_since_activity = calculated_metrics.get('days_since_activity', 0)
            return min(1.0, days_since_activity / param_value) if days_since_activity > param_value else 0.0
            
        elif param.key == 'stage_probability_misalignment_threshold':
            calculated_metrics = contextual_data.get('calculated_metrics', {})
            alignment = calculated_metrics.get('stage_probability_alignment', 1.0)
            misalignment = 1.0 - alignment
            return 1.0 if misalignment > (param_value / 100) else 0.0
        
        return 0.0
    
    def _calculate_rep_behavior_contribution(
        self, 
        param: ParameterDefinition, 
        param_value: Any, 
        opportunity: Dict[str, Any], 
        contextual_data: Dict[str, Any]
    ) -> float:
        """Calculate rep behavior parameter contribution"""
        
        rep_history = contextual_data.get('rep_history', {})
        
        if param.key == 'rep_historical_sandbagging_score':
            sandbagging_tendency = rep_history.get('sandbagging_tendency', 0.25)
            return sandbagging_tendency
            
        elif param.key == 'forecast_vs_actual_variance_threshold':
            forecast_accuracy = rep_history.get('forecast_accuracy', 0.85)
            variance = 1.0 - forecast_accuracy
            return 1.0 if variance > (param_value / 100) else 0.0
            
        elif param.key == 'deal_velocity_by_rep_threshold':
            avg_velocity = rep_history.get('avg_deal_velocity', 60)
            calculated_metrics = contextual_data.get('calculated_metrics', {})
            deal_age = calculated_metrics.get('deal_age_days', 30)
            current_velocity = 30 / max(1, deal_age / 30)  # Simplified velocity calculation
            return 1.0 if current_velocity < param_value else 0.0
            
        elif param.key == 'quarter_end_surge_multiplier':
            surge_pattern = rep_history.get('quarter_end_surge_pattern', 0.2)
            return surge_pattern * param_value
        
        return 0.0
    
    def _calculate_temporal_contribution(
        self, 
        param: ParameterDefinition, 
        param_value: Any, 
        opportunity: Dict[str, Any], 
        contextual_data: Dict[str, Any]
    ) -> float:
        """Calculate temporal pattern parameter contribution"""
        
        close_date = opportunity.get('CloseDate')
        if not close_date:
            return 0.0
        
        try:
            close_date = pd.to_datetime(close_date)
            now = datetime.now()
            
            if param.key == 'seasonal_sandbagging_weight':
                # Q4 has higher sandbagging risk
                is_q4 = close_date.month in [10, 11, 12]
                return param_value if is_q4 else 0.0
                
            elif param.key == 'month_end_behavior_multiplier':
                # Last 3 days of month
                days_in_month = close_date.days_in_month
                is_month_end = close_date.day > (days_in_month - 3)
                return 0.5 * param_value if is_month_end else 0.0
                
            elif param.key == 'fiscal_year_end_surge_threshold':
                # Assume fiscal year ends in December
                is_fiscal_year_end = close_date.month == 12
                return 0.6 if is_fiscal_year_end else 0.0
        
        except Exception as e:
            logger.warning(f"Error calculating temporal contribution: {e}")
            
        return 0.0
    
    def _calculate_organizational_contribution(
        self, 
        param: ParameterDefinition, 
        param_value: Any, 
        opportunity: Dict[str, Any], 
        contextual_data: Dict[str, Any]
    ) -> float:
        """Calculate organizational parameter contribution"""
        
        rep_history = contextual_data.get('rep_history', {})
        
        if param.key == 'quota_achievement_history_weight':
            quota_achievement = rep_history.get('quota_achievement_rate', 0.9)
            # Lower achievement may indicate sandbagging pressure
            if quota_achievement < 0.8:
                return param_value * 0.8
            elif quota_achievement > 1.2:
                return param_value * 0.3  # Overachievers may sandbag next period
            return 0.0
            
        elif param.key == 'team_performance_variance_threshold':
            # Simulated team performance comparison
            rep_performance = rep_history.get('quota_achievement_rate', 0.9)
            team_avg = 0.95  # Simulated team average
            variance = abs(rep_performance - team_avg) / team_avg
            return 1.0 if variance > (param_value / 100) else 0.0
        
        return 0.0
    
    def _calculate_market_contribution(
        self, 
        param: ParameterDefinition, 
        param_value: Any, 
        opportunity: Dict[str, Any], 
        contextual_data: Dict[str, Any]
    ) -> float:
        """Calculate market & customer parameter contribution"""
        
        market_data = contextual_data.get('market_data', {})
        
        if param.key == 'industry_benchmark_adjustment':
            competitive_intensity = market_data.get('competitive_intensity', 0.8)
            # Higher competition may lead to more conservative forecasting
            return competitive_intensity * param_value
            
        elif param.key == 'customer_size_segmentation_weight':
            amount = opportunity.get('Amount', 0)
            # Enterprise deals (>1M) have different patterns
            if amount > 1000000:
                return param_value * 0.7  # Enterprise deals often have longer cycles
            elif amount < 50000:
                return param_value * 0.3  # SMB deals are more predictable
            return 0.0
            
        elif param.key == 'competitive_landscape_intensity':
            competitive_intensity = market_data.get('competitive_intensity', 0.8)
            return competitive_intensity * param_value
        
        return 0.0
    
    def _get_parameter_weight(self, param: ParameterDefinition) -> float:
        """Get weight for parameter based on impact level"""
        impact_weights = {
            'critical': 1.0,
            'high': 0.8,
            'medium': 0.6,
            'low': 0.4
        }
        return impact_weights.get(param.impact_level, 0.6)
    
    def _calculate_composite_score(self, category_scores: Dict[ParameterCategory, float]) -> float:
        """Calculate weighted composite sandbagging score"""
        
        composite_score = 0.0
        total_weight = 0.0
        
        for category, score in category_scores.items():
            weight = self.category_weights.get(category, 0.1)
            composite_score += score * weight
            total_weight += weight
        
        # Normalize by total weight
        if total_weight > 0:
            composite_score = composite_score / total_weight
        
        return max(0.0, min(1.0, composite_score))
    
    def _determine_risk_level(self, overall_score: float) -> str:
        """Determine risk level based on overall score"""
        if overall_score >= self.risk_thresholds['CRITICAL']:
            return 'CRITICAL'
        elif overall_score >= self.risk_thresholds['HIGH']:
            return 'HIGH'
        elif overall_score >= self.risk_thresholds['MEDIUM']:
            return 'MEDIUM'
        else:
            return 'LOW'
    
    def _calculate_confidence(
        self, 
        active_parameters: List[ParameterDefinition], 
        contextual_data: Dict[str, Any]
    ) -> float:
        """Calculate confidence in the assessment"""
        
        confidence = 0.5  # Base confidence
        
        # More parameters = higher confidence
        param_count_boost = min(0.3, len(active_parameters) / 100 * 0.3)
        confidence += param_count_boost
        
        # Historical data availability boosts confidence
        if contextual_data.get('rep_history'):
            confidence += 0.15
        
        # Market data availability boosts confidence
        if contextual_data.get('market_data'):
            confidence += 0.1
        
        # Calculated metrics boost confidence
        if contextual_data.get('calculated_metrics'):
            confidence += 0.1
        
        return max(0.0, min(1.0, confidence))
    
    def _identify_contributing_factors(
        self, 
        category_scores: Dict[ParameterCategory, float], 
        active_parameters: List[ParameterDefinition]
    ) -> List[str]:
        """Identify main contributing factors to sandbagging risk"""
        
        factors = []
        
        # Category-based factors
        for category, score in category_scores.items():
            if score > 0.6:
                if category == ParameterCategory.DEAL_LEVEL:
                    factors.append("Deal characteristics indicate sandbagging risk")
                elif category == ParameterCategory.REP_BEHAVIOR:
                    factors.append("Rep behavioral patterns suggest sandbagging")
                elif category == ParameterCategory.TEMPORAL_PATTERNS:
                    factors.append("Timing patterns consistent with sandbagging")
                elif category == ParameterCategory.ORGANIZATIONAL:
                    factors.append("Organizational factors may encourage sandbagging")
                elif category == ParameterCategory.MARKET_CUSTOMER:
                    factors.append("Market conditions may influence conservative forecasting")
        
        # Parameter-specific factors
        high_impact_params = [p for p in active_parameters if p.impact_level in ['critical', 'high']]
        if len(high_impact_params) > 10:
            factors.append(f"Multiple high-impact risk factors detected ({len(high_impact_params)} parameters)")
        
        return factors[:5]  # Limit to top 5 factors
    
    def _generate_recommendations(
        self, 
        risk_level: str, 
        contributing_factors: List[str], 
        opportunity: Dict[str, Any]
    ) -> List[str]:
        """Generate actionable recommendations"""
        
        recommendations = []
        deal_name = opportunity.get('Name', 'this deal')
        
        if risk_level in ['CRITICAL', 'HIGH']:
            recommendations.append(f"URGENT: Immediate review required for {deal_name}")
            recommendations.append("Schedule 1-on-1 with rep to discuss deal status and probability accuracy")
            recommendations.append("Request detailed deal progression timeline and supporting evidence")
            
        elif risk_level == 'MEDIUM':
            recommendations.append(f"Monitor {deal_name} closely for probability updates")
            recommendations.append("Review deal activities and stakeholder engagement")
            
        else:
            recommendations.append(f"Continue standard monitoring of {deal_name}")
        
        # Factor-specific recommendations
        if "Rep behavioral patterns" in str(contributing_factors):
            recommendations.append("Consider additional coaching on forecast accuracy")
        
        if "Timing patterns" in str(contributing_factors):
            recommendations.append("Review deal timeline and quarter-end dynamics")
        
        return recommendations[:4]  # Limit to top 4 recommendations
    
    def _generate_next_actions(self, risk_level: str, opportunity: Dict[str, Any]) -> List[str]:
        """Generate specific next actions"""
        
        actions = []
        
        if risk_level == 'CRITICAL':
            actions.extend([
                "Escalate to sales management immediately",
                "Conduct deal review within 24 hours",
                "Verify customer engagement and buying signals"
            ])
        elif risk_level == 'HIGH':
            actions.extend([
                "Schedule deal review within 48 hours",
                "Request rep to provide deal evidence",
                "Contact customer to verify timeline"
            ])
        elif risk_level == 'MEDIUM':
            actions.extend([
                "Monitor for probability updates",
                "Review in weekly pipeline meeting"
            ])
        else:
            actions.append("Continue standard pipeline management")
        
        return actions
    
    def _identify_confidence_factors(
        self, 
        contextual_data: Dict[str, Any], 
        active_parameters: List[ParameterDefinition]
    ) -> List[str]:
        """Identify factors that affect confidence in the assessment"""
        
        factors = []
        
        if contextual_data.get('rep_history'):
            factors.append("Historical rep performance data available")
        
        if contextual_data.get('calculated_metrics'):
            factors.append("Advanced deal metrics calculated")
        
        if len(active_parameters) > 50:
            factors.append(f"Comprehensive analysis with {len(active_parameters)} parameters")
        
        if contextual_data.get('market_data'):
            factors.append("Market intelligence data integrated")
        
        return factors
    
    def _extract_parameter_values(
        self, 
        opportunity: Dict[str, Any], 
        contextual_data: Dict[str, Any], 
        active_parameters: List[ParameterDefinition]
    ) -> Dict[str, Any]:
        """Extract parameter values for transparency"""
        
        values = {}
        
        # Basic opportunity data
        values.update({
            'probability': opportunity.get('Probability', 0),
            'amount': opportunity.get('Amount', 0),
            'stage': opportunity.get('StageName', ''),
            'close_date': opportunity.get('CloseDate', ''),
            'owner_id': opportunity.get('OwnerId', '')
        })
        
        # Calculated metrics
        calculated_metrics = contextual_data.get('calculated_metrics', {})
        values.update(calculated_metrics)
        
        # Rep history data
        rep_history = contextual_data.get('rep_history', {})
        values.update({f"rep_{k}": v for k, v in rep_history.items()})
        
        # Market data
        market_data = contextual_data.get('market_data', {})
        values.update({f"market_{k}": v for k, v in market_data.items()})
        
        return values
    
    def _create_fallback_assessment(self, opportunity: Dict[str, Any]) -> SandbaggingAssessment:
        """Create fallback assessment when analysis fails"""
        
        opp_id = opportunity.get('Id', f"opp_{hash(str(opportunity))}")
        probability = opportunity.get('Probability', 50)
        
        # Simple fallback scoring
        fallback_score = max(0.0, (50 - probability) / 50) if probability < 50 else 0.0
        risk_level = 'HIGH' if fallback_score > 0.7 else 'MEDIUM' if fallback_score > 0.4 else 'LOW'
        
        return SandbaggingAssessment(
            opportunity_id=opp_id,
            overall_score=fallback_score,
            risk_level=risk_level,
            confidence=0.3,  # Low confidence for fallback
            deal_level_score=fallback_score,
            rep_behavior_score=0.0,
            temporal_score=0.0,
            organizational_score=0.0,
            market_score=0.0,
            active_parameters=['sandbagging_threshold'],
            parameter_values={'probability': probability},
            contributing_factors=['Limited data available for comprehensive analysis'],
            recommendations=[f"Basic analysis suggests {risk_level.lower()} sandbagging risk"],
            next_actions=['Gather more data for comprehensive analysis'],
            analysis_timestamp=datetime.now().isoformat(),
            confidence_factors=['Fallback analysis - limited data available']
        )

# Global instance
comprehensive_analyzer = ComprehensiveSandbaggingAnalyzer()
