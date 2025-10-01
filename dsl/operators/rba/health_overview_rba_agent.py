"""
Health Overview RBA Agent
Single-purpose, focused RBA agent for comprehensive pipeline health overview

This agent ONLY handles comprehensive pipeline health assessment.
Lightweight, modular, configuration-driven.
"""

import logging
from typing import Dict, List, Any
from datetime import datetime, timedelta
from collections import defaultdict

from .enhanced_base_rba_agent import EnhancedBaseRBAAgent

logger = logging.getLogger(__name__)

class HealthOverviewRBAAgent(EnhancedBaseRBAAgent):
    """
    Single-purpose RBA agent for health overview
    
    Features:
    - ONLY handles comprehensive health overview
    - Configuration-driven health metrics
    - Lightweight and focused
    - Multi-dimensional health assessment
    - Clear separation of concerns
    """
    
    # Agent metadata for dynamic discovery
    AGENT_TYPE = "RBA"
    AGENT_NAME = "health_overview"
    AGENT_DESCRIPTION = "Comprehensive pipeline health overview combining multiple health dimensions"
    SUPPORTED_ANALYSIS_TYPES = ["health_overview", "pipeline_health"]
    
    def __init__(self, config: Dict[str, Any] = None):
        super().__init__(config or {})
        
        # Health overview specific defaults
        self.default_config = {
            'health_dimensions': [
                'stage_distribution',
                'velocity_health',
                'activity_health',
                'data_quality_health',
                'coverage_health'
            ],
            'overall_health_weights': {
                'stage_distribution': 0.2,
                'velocity_health': 0.2,
                'activity_health': 0.2,
                'data_quality_health': 0.2,
                'coverage_health': 0.2
            },
            'health_score_thresholds': {
                'excellent': 90,
                'good': 75,
                'fair': 60,
                'poor': 40
            }
        }
    
    async def execute(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Execute comprehensive health overview analysis"""
        
        try:
            opportunities = input_data.get('opportunities', [])
            config = {**self.default_config, **input_data.get('config', {})}
            
            logger.info(f"🎯 Health Overview RBA: Analyzing {len(opportunities)} opportunities")
            
            # Calculate health dimensions
            health_dimensions = {}
            dimension_list = config.get('health_dimensions', [])
            
            for dimension in dimension_list:
                health_dimensions[dimension] = self._calculate_health_dimension(opportunities, dimension, config)
            
            # Calculate overall health score
            overall_health = self._calculate_overall_health_score(health_dimensions, config)
            
            # Generate health insights
            health_insights = self._generate_health_insights(health_dimensions, overall_health, config)
            
            # Identify health issues
            health_issues = self._identify_health_issues(health_dimensions, config)
            
            # Generate improvement recommendations
            recommendations = self._generate_health_recommendations(health_dimensions, overall_health, config)
            
            return {
                'success': True,
                'agent_name': self.AGENT_NAME,
                'analysis_type': 'health_overview',
                'total_opportunities': len(opportunities),
                'overall_health_score': overall_health['score'],
                'overall_health_grade': overall_health['grade'],
                'health_dimensions': health_dimensions,
                'health_insights': health_insights,
                'health_issues': health_issues,
                'recommendations': recommendations,
                'configuration_used': config,
                'analysis_timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"❌ Health Overview RBA failed: {e}")
            return {
                'success': False,
                'error': str(e),
                'agent_name': self.AGENT_NAME,
                'analysis_timestamp': datetime.now().isoformat()
            }
    
    def _calculate_health_dimension(self, opportunities: List[Dict[str, Any]], dimension: str, config: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate health score for a specific dimension"""
        
        if dimension == 'stage_distribution':
            return self._calculate_stage_distribution_health(opportunities)
        
        elif dimension == 'velocity_health':
            return self._calculate_velocity_health(opportunities)
        
        elif dimension == 'activity_health':
            return self._calculate_activity_health(opportunities)
        
        elif dimension == 'data_quality_health':
            return self._calculate_data_quality_health(opportunities)
        
        elif dimension == 'coverage_health':
            return self._calculate_coverage_health(opportunities)
        
        else:
            return {'score': 50, 'status': 'UNKNOWN', 'details': {}}
    
    def _calculate_stage_distribution_health(self, opportunities: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Calculate stage distribution health"""
        
        open_deals = [opp for opp in opportunities if 'closed' not in opp.get('StageName', '').lower()]
        
        if not open_deals:
            return {'score': 0, 'status': 'NO_DATA', 'details': {}}
        
        # Count deals by stage
        stage_counts = defaultdict(int)
        for opp in open_deals:
            stage = opp.get('StageName', 'Unknown')
            stage_counts[stage] += 1
        
        # Calculate distribution health based on funnel balance
        total_deals = len(open_deals)
        early_stage_count = 0
        late_stage_count = 0
        
        for stage, count in stage_counts.items():
            stage_lower = stage.lower()
            if any(keyword in stage_lower for keyword in ['lead', 'prospect', 'qualification', 'discovery']):
                early_stage_count += count
            elif any(keyword in stage_lower for keyword in ['proposal', 'negotiation', 'contract', 'closing']):
                late_stage_count += count
        
        # Healthy distribution should have more early stage deals
        early_stage_ratio = early_stage_count / total_deals if total_deals > 0 else 0
        late_stage_ratio = late_stage_count / total_deals if total_deals > 0 else 0
        
        # Score based on balanced funnel (more early, some late)
        if 0.4 <= early_stage_ratio <= 0.7 and 0.2 <= late_stage_ratio <= 0.4:
            score = 90
            status = 'EXCELLENT'
        elif early_stage_ratio >= 0.3 and late_stage_ratio >= 0.1:
            score = 75
            status = 'GOOD'
        else:
            score = 50
            status = 'FAIR'
        
        return {
            'score': score,
            'status': status,
            'details': {
                'total_open_deals': total_deals,
                'early_stage_ratio': round(early_stage_ratio, 2),
                'late_stage_ratio': round(late_stage_ratio, 2),
                'stage_distribution': dict(stage_counts)
            }
        }
    
    def _calculate_velocity_health(self, opportunities: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Calculate velocity health"""
        
        open_deals = [opp for opp in opportunities if 'closed' not in opp.get('StageName', '').lower()]
        
        if not open_deals:
            return {'score': 0, 'status': 'NO_DATA', 'details': {}}
        
        # Calculate average days in current stage
        total_days = 0
        deals_with_stage_date = 0
        stale_deals = 0
        
        for opp in open_deals:
            stage_date_str = opp.get('StageChangeDate') or opp.get('LastModifiedDate')
            if stage_date_str:
                try:
                    stage_date = datetime.strptime(stage_date_str.split('T')[0], '%Y-%m-%d')
                    days_in_stage = (datetime.now() - stage_date).days
                    total_days += days_in_stage
                    deals_with_stage_date += 1
                    
                    if days_in_stage > 60:  # Consider 60+ days as stale
                        stale_deals += 1
                except:
                    pass
        
        if deals_with_stage_date == 0:
            return {'score': 0, 'status': 'NO_DATA', 'details': {}}
        
        avg_days_in_stage = total_days / deals_with_stage_date
        stale_percentage = (stale_deals / len(open_deals)) * 100
        
        # Score based on velocity
        if avg_days_in_stage <= 30 and stale_percentage <= 10:
            score = 90
            status = 'EXCELLENT'
        elif avg_days_in_stage <= 45 and stale_percentage <= 20:
            score = 75
            status = 'GOOD'
        elif avg_days_in_stage <= 60 and stale_percentage <= 35:
            score = 60
            status = 'FAIR'
        else:
            score = 40
            status = 'POOR'
        
        return {
            'score': score,
            'status': status,
            'details': {
                'average_days_in_stage': round(avg_days_in_stage, 1),
                'stale_deals_count': stale_deals,
                'stale_percentage': round(stale_percentage, 1)
            }
        }
    
    def _calculate_activity_health(self, opportunities: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Calculate activity health"""
        
        open_deals = [opp for opp in opportunities if 'closed' not in opp.get('StageName', '').lower()]
        
        if not open_deals:
            return {'score': 0, 'status': 'NO_DATA', 'details': {}}
        
        # Calculate activity recency
        recent_activity_deals = 0
        stale_activity_deals = 0
        no_activity_deals = 0
        
        for opp in open_deals:
            activity_date_str = opp.get('ActivityDate')
            if activity_date_str:
                try:
                    activity_date = datetime.strptime(activity_date_str.split('T')[0], '%Y-%m-%d')
                    days_since_activity = (datetime.now() - activity_date).days
                    
                    if days_since_activity <= 14:
                        recent_activity_deals += 1
                    elif days_since_activity <= 30:
                        # Moderate activity
                        pass
                    else:
                        stale_activity_deals += 1
                except:
                    no_activity_deals += 1
            else:
                no_activity_deals += 1
        
        recent_activity_percentage = (recent_activity_deals / len(open_deals)) * 100
        stale_activity_percentage = ((stale_activity_deals + no_activity_deals) / len(open_deals)) * 100
        
        # Score based on activity recency
        if recent_activity_percentage >= 70 and stale_activity_percentage <= 15:
            score = 90
            status = 'EXCELLENT'
        elif recent_activity_percentage >= 50 and stale_activity_percentage <= 30:
            score = 75
            status = 'GOOD'
        elif recent_activity_percentage >= 30 and stale_activity_percentage <= 50:
            score = 60
            status = 'FAIR'
        else:
            score = 40
            status = 'POOR'
        
        return {
            'score': score,
            'status': status,
            'details': {
                'recent_activity_deals': recent_activity_deals,
                'recent_activity_percentage': round(recent_activity_percentage, 1),
                'stale_activity_deals': stale_activity_deals + no_activity_deals,
                'stale_activity_percentage': round(stale_activity_percentage, 1)
            }
        }
    
    def _calculate_data_quality_health(self, opportunities: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Calculate data quality health"""
        
        if not opportunities:
            return {'score': 0, 'status': 'NO_DATA', 'details': {}}
        
        required_fields = ['Name', 'Amount', 'CloseDate', 'StageName', 'OwnerId']
        total_fields_checked = 0
        missing_fields_count = 0
        
        for opp in opportunities:
            for field in required_fields:
                total_fields_checked += 1
                value = opp.get(field)
                if value is None or value == '' or (isinstance(value, str) and value.strip() == ''):
                    missing_fields_count += 1
        
        data_completeness_percentage = ((total_fields_checked - missing_fields_count) / total_fields_checked) * 100
        
        # Score based on data completeness
        if data_completeness_percentage >= 95:
            score = 90
            status = 'EXCELLENT'
        elif data_completeness_percentage >= 85:
            score = 75
            status = 'GOOD'
        elif data_completeness_percentage >= 70:
            score = 60
            status = 'FAIR'
        else:
            score = 40
            status = 'POOR'
        
        return {
            'score': score,
            'status': status,
            'details': {
                'data_completeness_percentage': round(data_completeness_percentage, 1),
                'missing_fields_count': missing_fields_count,
                'total_fields_checked': total_fields_checked
            }
        }
    
    def _calculate_coverage_health(self, opportunities: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Calculate coverage health"""
        
        open_deals = [opp for opp in opportunities if 'closed' not in opp.get('StageName', '').lower()]
        
        if not open_deals:
            return {'score': 0, 'status': 'NO_DATA', 'details': {}}
        
        # Calculate pipeline value
        total_pipeline = sum(self._safe_float(opp.get('Amount')) for opp in open_deals)
        weighted_pipeline = sum(
            self._safe_float(opp.get('Amount')) * (self._safe_float(opp.get('Probability')) / 100)
            for opp in open_deals
        )
        
        # Simulate quota (in real system, would come from actual quota data)
        estimated_quota = total_pipeline * 0.6  # Assume quota is 60% of pipeline
        coverage_ratio = (weighted_pipeline / estimated_quota) if estimated_quota > 0 else 0
        
        # Score based on coverage ratio
        if coverage_ratio >= 3.0:
            score = 90
            status = 'EXCELLENT'
        elif coverage_ratio >= 2.0:
            score = 75
            status = 'GOOD'
        elif coverage_ratio >= 1.0:
            score = 60
            status = 'FAIR'
        else:
            score = 40
            status = 'POOR'
        
        return {
            'score': score,
            'status': status,
            'details': {
                'total_pipeline': round(total_pipeline, 2),
                'weighted_pipeline': round(weighted_pipeline, 2),
                'estimated_quota': round(estimated_quota, 2),
                'coverage_ratio': round(coverage_ratio, 2)
            }
        }
    
    def _calculate_overall_health_score(self, health_dimensions: Dict[str, Any], config: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate overall health score"""
        
        weights = config.get('overall_health_weights', {})
        total_weighted_score = 0
        total_weights = 0
        
        for dimension, health_data in health_dimensions.items():
            weight = weights.get(dimension, 0.2)  # Default equal weight
            score = health_data.get('score', 0)
            
            total_weighted_score += score * weight
            total_weights += weight
        
        overall_score = total_weighted_score / total_weights if total_weights > 0 else 0
        
        # Determine grade
        thresholds = config.get('health_score_thresholds', {})
        if overall_score >= thresholds.get('excellent', 90):
            grade = 'EXCELLENT'
        elif overall_score >= thresholds.get('good', 75):
            grade = 'GOOD'
        elif overall_score >= thresholds.get('fair', 60):
            grade = 'FAIR'
        elif overall_score >= thresholds.get('poor', 40):
            grade = 'POOR'
        else:
            grade = 'CRITICAL'
        
        return {
            'score': round(overall_score, 1),
            'grade': grade,
            'weights_used': weights
        }
    
    def _generate_health_insights(
        self, 
        health_dimensions: Dict[str, Any], 
        overall_health: Dict[str, Any], 
        config: Dict[str, Any]
    ) -> List[str]:
        """Generate health insights"""
        
        insights = []
        
        # Overall health insight
        insights.append(f"Overall pipeline health: {overall_health['grade']} ({overall_health['score']}/100)")
        
        # Dimension-specific insights
        for dimension, health_data in health_dimensions.items():
            status = health_data.get('status', 'UNKNOWN')
            score = health_data.get('score', 0)
            
            if status in ['POOR', 'CRITICAL']:
                insights.append(f"{dimension.replace('_', ' ').title()}: Needs immediate attention ({score}/100)")
            elif status == 'EXCELLENT':
                insights.append(f"{dimension.replace('_', ' ').title()}: Performing excellently ({score}/100)")
        
        return insights
    
    def _identify_health_issues(self, health_dimensions: Dict[str, Any], config: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Identify specific health issues"""
        
        issues = []
        
        for dimension, health_data in health_dimensions.items():
            status = health_data.get('status', 'UNKNOWN')
            score = health_data.get('score', 0)
            
            if status in ['POOR', 'CRITICAL', 'FAIR']:
                severity = 'HIGH' if status in ['POOR', 'CRITICAL'] else 'MEDIUM'
                
                issues.append({
                    'dimension': dimension,
                    'dimension_name': dimension.replace('_', ' ').title(),
                    'score': score,
                    'status': status,
                    'severity': severity,
                    'risk_level': severity,
                    'issue_type': 'HEALTH_ISSUE',
                    'priority': 'HIGH' if severity == 'HIGH' else 'MEDIUM',
                    'description': f"{dimension.replace('_', ' ').title()} health is {status.lower()} ({score}/100)",
                    'recommended_action': self._get_dimension_recommendation(dimension, status),
                    'details': health_data.get('details', {}),
                    'analysis_timestamp': datetime.now().isoformat()
                })
        
        return issues
    
    def _get_dimension_recommendation(self, dimension: str, status: str) -> str:
        """Get recommendation for specific dimension"""
        
        recommendations = {
            'stage_distribution': 'Rebalance pipeline funnel - increase early stage prospecting',
            'velocity_health': 'Accelerate deal progression - review stale deals and stage advancement',
            'activity_health': 'Increase customer engagement - schedule more activities on open deals',
            'data_quality_health': 'Improve data completeness - update missing required fields',
            'coverage_health': 'Increase pipeline generation - build more qualified opportunities'
        }
        
        base_rec = recommendations.get(dimension, 'Review and improve this health dimension')
        
        if status in ['POOR', 'CRITICAL']:
            return f"URGENT: {base_rec}"
        else:
            return base_rec
    
    def _generate_health_recommendations(
        self, 
        health_dimensions: Dict[str, Any], 
        overall_health: Dict[str, Any], 
        config: Dict[str, Any]
    ) -> List[str]:
        """Generate health improvement recommendations"""
        
        recommendations = []
        
        # Overall recommendations based on grade
        grade = overall_health['grade']
        if grade in ['POOR', 'CRITICAL']:
            recommendations.append("Immediate pipeline health intervention required - focus on top 3 weakest areas")
        elif grade == 'FAIR':
            recommendations.append("Pipeline health needs improvement - prioritize underperforming dimensions")
        
        # Specific recommendations for poor dimensions
        poor_dimensions = [
            dim for dim, data in health_dimensions.items()
            if data.get('status') in ['POOR', 'CRITICAL']
        ]
        
        if poor_dimensions:
            recommendations.append(f"Focus improvement efforts on: {', '.join(d.replace('_', ' ') for d in poor_dimensions)}")
        
        return recommendations
    
    @classmethod
    def get_agent_metadata(cls) -> Dict[str, Any]:
        """Get agent metadata for registry"""
        return {
            'agent_type': cls.AGENT_TYPE,
            'agent_name': cls.AGENT_NAME,
            'agent_description': cls.AGENT_DESCRIPTION,
            'supported_analysis_types': cls.SUPPORTED_ANALYSIS_TYPES,
            'class_name': cls.__name__,
            'module_path': cls.__module__
        }
