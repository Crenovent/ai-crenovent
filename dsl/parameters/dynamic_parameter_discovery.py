"""
Dynamic Parameter Discovery System
Unified parameter generation for all RBA workflows based on available data and business context
"""

import json
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from enum import Enum
import logging

logger = logging.getLogger(__name__)

class ParameterType(Enum):
    """Parameter types for dynamic discovery"""
    THRESHOLD = "threshold"
    AMOUNT = "amount"  
    PERCENTAGE = "percentage"
    DAYS = "days"
    COUNT = "count"
    MULTIPLIER = "multiplier"
    BOOLEAN = "boolean"
    LIST = "list"
    TEXT = "text"
    DATE = "date"
    RATIO = "ratio"
    SCORE = "score"
    WEIGHT = "weight"
    FREQUENCY = "frequency"
    VELOCITY = "velocity"
    PATTERN = "pattern"
    CORRELATION = "correlation"

class ParameterCategory(Enum):
    """Parameter categories for organization"""
    # Core Categories (existing)
    RISK_SCORING = "risk_scoring"
    TIME_BASED = "time_based"
    FINANCIAL = "financial"
    BEHAVIORAL = "behavioral"
    STAGE_BASED = "stage_based"
    GEOGRAPHIC = "geographic"
    INDUSTRY = "industry"
    ACTIVITY = "activity"
    
    # Comprehensive Categories (new)
    DEAL_LEVEL = "deal_level"
    REP_BEHAVIOR = "rep_behavior"
    TEMPORAL_PATTERNS = "temporal_patterns"
    ORGANIZATIONAL = "organizational"
    MARKET_CUSTOMER = "market_customer"
    CRM_DATA_QUALITY = "crm_data_quality"
    ENGAGEMENT_METRICS = "engagement_metrics"
    COMPETITIVE_INTELLIGENCE = "competitive_intelligence"
    STAKEHOLDER_ANALYSIS = "stakeholder_analysis"
    PIPELINE_VELOCITY = "pipeline_velocity"
    HISTORICAL_PATTERNS = "historical_patterns"
    FORECAST_ACCURACY = "forecast_accuracy"
    COMPENSATION_ALIGNMENT = "compensation_alignment"
    SEASONAL_FACTORS = "seasonal_factors"
    ECONOMIC_INDICATORS = "economic_indicators"

@dataclass
class ParameterDefinition:
    """Dynamic parameter definition"""
    key: str
    label: str
    description: str
    type: ParameterType
    category: ParameterCategory
    default_value: Any
    min_value: Optional[Any] = None
    max_value: Optional[Any] = None
    step: Optional[float] = None
    options: Optional[List[str]] = None
    depends_on_fields: Optional[List[str]] = None
    business_context: Optional[str] = None
    expert_mode_only: bool = False
    
    # Enhanced attributes for comprehensive detection
    data_source: Optional[str] = None  # 'crm', 'historical', 'external', 'calculated'
    calculation_method: Optional[str] = None  # How to derive this parameter
    requires_historical_data: bool = False
    requires_external_data: bool = False
    impact_level: str = "medium"  # 'low', 'medium', 'high', 'critical'
    reliability_score: float = 1.0  # 0.0 to 1.0
    related_parameters: Optional[List[str]] = None
    validation_rules: Optional[List[str]] = None

class DynamicParameterDiscovery:
    """
    Unified parameter discovery system for all RBA workflows
    Dynamically generates parameters based on:
    1. Available CSV fields
    2. Workflow type 
    3. Business context
    4. User expertise level
    """
    
    def __init__(self):
        self.available_fields = self._get_csv_field_schema()
        self.workflow_contexts = self._get_workflow_contexts()
    
    def _get_csv_field_schema(self) -> Dict[str, Dict]:
        """Define available CSV fields and their capabilities"""
        return {
            # Core Opportunity Fields
            'Amount': {
                'type': 'financial',
                'enables_parameters': ['amount_thresholds', 'value_based_scoring', 'deal_size_analysis']
            },
            'Probability': {
                'type': 'percentage', 
                'enables_parameters': ['probability_thresholds', 'confidence_scoring', 'sandbagging_detection']
            },
            'CloseDate': {
                'type': 'date',
                'enables_parameters': ['time_based_analysis', 'urgency_scoring', 'seasonal_patterns']
            },
            'StageName': {
                'type': 'categorical',
                'enables_parameters': ['stage_based_rules', 'velocity_analysis', 'conversion_tracking']
            },
            'CreatedDate': {
                'type': 'date',
                'enables_parameters': ['age_analysis', 'lifecycle_tracking', 'vintage_scoring']
            },
            'ActivityDate': {
                'type': 'date', 
                'enables_parameters': ['engagement_analysis', 'stale_detection', 'momentum_tracking']
            },
            
            # Account Intelligence Fields
            'Industry': {
                'type': 'categorical',
                'enables_parameters': ['industry_benchmarks', 'vertical_rules', 'sector_analysis']
            },
            'AnnualRevenue': {
                'type': 'financial',
                'enables_parameters': ['account_size_rules', 'enterprise_vs_smb', 'revenue_correlation']
            },
            'NumberOfEmployees': {
                'type': 'numeric',
                'enables_parameters': ['company_size_rules', 'headcount_analysis', 'scaling_indicators']
            },
            
            # Geographic Fields  
            'BillingCountry': {
                'type': 'categorical',
                'enables_parameters': ['geo_rules', 'regional_analysis', 'compliance_zones']
            },
            'BillingState': {
                'type': 'categorical', 
                'enables_parameters': ['territory_rules', 'state_regulations', 'local_patterns']
            },
            
            # Engagement Fields
            'LeadSource': {
                'type': 'categorical',
                'enables_parameters': ['source_quality', 'channel_analysis', 'attribution_rules']
            },
            'Priority': {
                'type': 'categorical',
                'enables_parameters': ['priority_weighting', 'urgency_rules', 'escalation_triggers']
            },
            'Status': {
                'type': 'categorical',
                'enables_parameters': ['status_based_rules', 'workflow_states', 'progression_analysis']
            }
        }
    
    def _get_workflow_contexts(self) -> Dict[str, Dict]:
        """Define business contexts for different workflows"""
        return {
            'sandbagging_detection': {
                'primary_goal': 'Identify deals with artificially low probability vs actual likelihood',
                'key_indicators': ['probability', 'amount', 'stage', 'activity_recency'],
                'business_impact': 'forecast_accuracy',
                'stakeholders': ['sales_managers', 'cro', 'revenue_ops'],
                'complexity_factors': ['deal_size', 'stage_advancement', 'historical_patterns', 'rep_behavior']
            },
            'stage_velocity_analysis': {
                'primary_goal': 'Identify bottlenecks and slow-moving deals in pipeline',
                'key_indicators': ['stage_duration', 'progression_rate', 'activity_frequency'],
                'business_impact': 'sales_efficiency',
                'stakeholders': ['sales_managers', 'reps', 'sales_ops'],
                'complexity_factors': ['stage_complexity', 'deal_size_impact', 'seasonal_patterns']
            },
            'pipeline_health_overview': {
                'primary_goal': 'Comprehensive pipeline health assessment',
                'key_indicators': ['coverage_ratio', 'risk_distribution', 'stage_health'],
                'business_impact': 'revenue_predictability', 
                'stakeholders': ['cro', 'vp_sales', 'board'],
                'complexity_factors': ['multi_dimensional_analysis', 'predictive_modeling', 'risk_aggregation']
            }
        }
    
    def discover_parameters(self, workflow_type: str, expertise_level: str = 'expert') -> List[ParameterDefinition]:
        """
        Dynamically discover parameters for a workflow
        
        Args:
            workflow_type: Type of workflow (sandbagging_detection, etc.)
            expertise_level: User expertise (basic, advanced, expert)
        """
        logger.info(f"🔍 Discovering parameters for {workflow_type} at {expertise_level} level")
        
        if workflow_type not in self.workflow_contexts:
            raise ValueError(f"Unknown workflow type: {workflow_type}")
        
        context = self.workflow_contexts[workflow_type]
        parameters = []
        
        # Generate parameters based on workflow type
        if workflow_type == 'sandbagging_detection':
            parameters.extend(self._generate_sandbagging_parameters(expertise_level))
        elif workflow_type == 'stage_velocity_analysis':
            parameters.extend(self._generate_velocity_parameters(expertise_level))
        elif workflow_type == 'pipeline_health_overview':
            parameters.extend(self._generate_health_parameters(expertise_level))
        
        # Add universal parameters available to all workflows
        parameters.extend(self._generate_universal_parameters(expertise_level))
        
        logger.info(f"✅ Discovered {len(parameters)} parameters for {workflow_type}")
        return parameters
    
    def _generate_sandbagging_parameters(self, expertise_level: str) -> List[ParameterDefinition]:
        """Generate comprehensive sandbagging parameters (100+ factors)"""
        params = []
        
        # DEAL-LEVEL PARAMETERS (25)
        params.extend(self._get_deal_level_parameters())
        
        # REP BEHAVIOR PARAMETERS (25)  
        params.extend(self._get_rep_behavior_parameters())
        
        # TEMPORAL PARAMETERS (15)
        params.extend(self._get_temporal_parameters())
        
        # ORGANIZATIONAL PARAMETERS (15)
        params.extend(self._get_organizational_parameters())
        
        # MARKET & CUSTOMER PARAMETERS (20)
        params.extend(self._get_market_customer_parameters())
        
        # Filter by expertise level
        if expertise_level == "basic":
            params = [p for p in params if not p.expert_mode_only and p.impact_level in ["high", "critical"]]
        elif expertise_level == "advanced":
            params = [p for p in params if p.impact_level in ["medium", "high", "critical"]]
        # expert level gets all parameters
        
        logger.info(f"🎯 Generated {len(params)} sandbagging parameters for {expertise_level} level")
        return params
    
    def _generate_velocity_parameters(self, expertise_level: str) -> List[ParameterDefinition]:
        """Generate velocity analysis parameters"""
        params = []
        
        # Core velocity parameters would go here
        # Implementation similar to sandbagging but for velocity-specific needs
        
        return params
    
    def _generate_health_parameters(self, expertise_level: str) -> List[ParameterDefinition]:
        """Generate health overview parameters"""
        params = []
        
        # Core health parameters would go here
        # Implementation similar to sandbagging but for health-specific needs
        
        return params
    
    def _generate_universal_parameters(self, expertise_level: str) -> List[ParameterDefinition]:
        """Generate parameters available to all workflows"""
        params = []
        
        if expertise_level == 'expert':
            params.extend([
                ParameterDefinition(
                    key="data_freshness_threshold",
                    label="Data Freshness Threshold (hours)",
                    description="Maximum age of data to be considered fresh",
                    type=ParameterType.COUNT,
                    category=ParameterCategory.TIME_BASED,
                    default_value=24,
                    min_value=1,
                    max_value=168,
                    business_context="Stale data can lead to incorrect analysis",
                    expert_mode_only=True
                ),
                ParameterDefinition(
                    key="confidence_threshold",
                    label="Analysis Confidence Threshold (%)",
                    description="Minimum confidence level required for recommendations",
                    type=ParameterType.PERCENTAGE,
                    category=ParameterCategory.RISK_SCORING,
                    default_value=85,
                    min_value=50,
                    max_value=99,
                    business_context="Higher confidence reduces false positives but may miss edge cases",
                    expert_mode_only=True
                )
            ])
        
        return params
    
    def get_parameter_schema_for_frontend(self, workflow_type: str, expertise_level: str = 'expert') -> Dict[str, Any]:
        """
        Generate frontend-compatible parameter schema
        """
        parameters = self.discover_parameters(workflow_type, expertise_level)
        
        schema = {
            'workflow_type': workflow_type,
            'expertise_level': expertise_level,
            'parameter_count': len(parameters),
            'categories': {},
            'parameters': []
        }
        
        # Group by categories
        for param in parameters:
            category = param.category.value
            if category not in schema['categories']:
                schema['categories'][category] = []
            schema['categories'][category].append(param.key)
            
            # Convert to frontend format
            param_dict = {
                'key': param.key,
                'label': param.label,
                'description': param.description,
                'type': param.type.value,
                'category': category,
                'default': param.default_value,
                'expert_only': param.expert_mode_only
            }
            
            # Add type-specific properties
            if param.min_value is not None:
                param_dict['min'] = param.min_value
            if param.max_value is not None:
                param_dict['max'] = param.max_value
            if param.step is not None:
                param_dict['step'] = param.step
            if param.options is not None:
                param_dict['options'] = param.options
            if param.business_context:
                param_dict['help_text'] = param.business_context
            
            schema['parameters'].append(param_dict)
        
        return schema

    # ===== COMPREHENSIVE SANDBAGGING PARAMETER GENERATORS =====
    
    def _get_deal_level_parameters(self) -> List[ParameterDefinition]:
        """Generate 25 deal-level sandbagging parameters"""
        return [
            # Core Deal Parameters
            ParameterDefinition(
                key="sandbagging_threshold",
                label="Sandbagging Probability Threshold (%)",
                description="Deals below this probability are flagged for sandbagging analysis",
                type=ParameterType.PERCENTAGE,
                category=ParameterCategory.DEAL_LEVEL,
                default_value=30,
                min_value=5,
                max_value=80,
                depends_on_fields=['Probability'],
                business_context="Lower values catch more conservative sandbagging, higher values focus on obvious cases",
                data_source="crm",
                impact_level="critical"
            ),
            ParameterDefinition(
                key="high_value_threshold",
                label="High Value Deal Threshold ($)",
                description="Deal amount threshold for high-value sandbagging analysis",
                type=ParameterType.AMOUNT,
                category=ParameterCategory.DEAL_LEVEL,
                default_value=100000,
                min_value=10000,
                max_value=10000000,
                step=10000,
                depends_on_fields=['Amount'],
                business_context="Larger deals have higher sandbagging risk due to forecast impact",
                data_source="crm",
                impact_level="high"
            ),
            ParameterDefinition(
                key="advanced_stage_multiplier",
                label="Advanced Stage Risk Multiplier",
                description="Multiplier for sandbagging risk when deal is in advanced stages",
                type=ParameterType.MULTIPLIER,
                category=ParameterCategory.DEAL_LEVEL,
                default_value=1.5,
                min_value=1.0,
                max_value=3.0,
                step=0.1,
                depends_on_fields=['StageName'],
                business_context="Deals in late stages with low probability are highly suspicious",
                data_source="crm",
                impact_level="high"
            ),
            ParameterDefinition(
                key="stage_probability_misalignment_threshold",
                label="Stage-Probability Misalignment Threshold",
                description="Maximum acceptable gap between stage advancement and probability",
                type=ParameterType.PERCENTAGE,
                category=ParameterCategory.DEAL_LEVEL,
                default_value=40,
                min_value=10,
                max_value=70,
                depends_on_fields=['StageName', 'Probability'],
                business_context="Advanced stages should have higher probabilities",
                data_source="crm",
                impact_level="high",
                expert_mode_only=True
            ),
            ParameterDefinition(
                key="deal_age_vs_probability_factor",
                label="Deal Age vs Probability Factor",
                description="Weight for flagging old deals with low probabilities",
                type=ParameterType.WEIGHT,
                category=ParameterCategory.DEAL_LEVEL,
                default_value=0.6,
                min_value=0.1,
                max_value=1.0,
                step=0.1,
                depends_on_fields=['CreatedDate', 'Probability'],
                business_context="Old deals should have higher probabilities if they're still active",
                data_source="crm",
                impact_level="medium",
                expert_mode_only=True
            ),
            ParameterDefinition(
                key="close_date_pushback_threshold",
                label="Close Date Pushback Threshold",
                description="Number of date changes that trigger sandbagging alert",
                type=ParameterType.COUNT,
                category=ParameterCategory.DEAL_LEVEL,
                default_value=3,
                min_value=1,
                max_value=10,
                depends_on_fields=['CloseDate'],
                business_context="Repeated close date changes may indicate sandbagging",
                data_source="historical",
                requires_historical_data=True,
                impact_level="high",
                expert_mode_only=True
            ),
            ParameterDefinition(
                key="activity_recency_threshold_days",
                label="Activity Recency Threshold (Days)",
                description="Days since last activity before flagging as stale",
                type=ParameterType.DAYS,
                category=ParameterCategory.DEAL_LEVEL,
                default_value=14,
                min_value=3,
                max_value=90,
                depends_on_fields=['ActivityDate'],
                business_context="Lack of recent activity on 'active' deals suggests sandbagging",
                data_source="crm",
                impact_level="medium"
            ),
            ParameterDefinition(
                key="competitor_presence_risk_multiplier",
                label="Competitor Presence Risk Multiplier",
                description="Risk multiplier when competitors are present",
                type=ParameterType.MULTIPLIER,
                category=ParameterCategory.DEAL_LEVEL,
                default_value=1.3,
                min_value=1.0,
                max_value=2.0,
                step=0.1,
                depends_on_fields=['CompetitorAnalysis'],
                business_context="Competitive deals have higher uncertainty, may be sandbagged",
                data_source="crm",
                impact_level="medium",
                expert_mode_only=True
            ),
            ParameterDefinition(
                key="decision_maker_engagement_threshold",
                label="Decision Maker Engagement Threshold",
                description="Minimum engagement level required from decision makers",
                type=ParameterType.SCORE,
                category=ParameterCategory.DEAL_LEVEL,
                default_value=70,
                min_value=30,
                max_value=100,
                depends_on_fields=['StakeholderEngagement'],
                business_context="Low decision maker engagement with high probability is suspicious",
                data_source="calculated",
                calculation_method="engagement_score_from_activities",
                impact_level="medium",
                expert_mode_only=True
            ),
            ParameterDefinition(
                key="proposal_submitted_probability_gap",
                label="Proposal Submitted Probability Gap",
                description="Expected probability increase after proposal submission",
                type=ParameterType.PERCENTAGE,
                category=ParameterCategory.DEAL_LEVEL,
                default_value=20,
                min_value=10,
                max_value=50,
                depends_on_fields=['ProposalDate', 'Probability'],
                business_context="Probability should increase significantly after proposal",
                data_source="crm",
                impact_level="high",
                expert_mode_only=True
            ),
            ParameterDefinition(
                key="contract_sent_probability_minimum",
                label="Contract Sent Minimum Probability",
                description="Minimum probability expected when contract is sent",
                type=ParameterType.PERCENTAGE,
                category=ParameterCategory.DEAL_LEVEL,
                default_value=75,
                min_value=50,
                max_value=95,
                depends_on_fields=['ContractDate', 'Probability'],
                business_context="Sending contract implies high confidence in closure",
                data_source="crm",
                impact_level="high",
                expert_mode_only=True
            ),
            ParameterDefinition(
                key="budget_confirmed_probability_boost",
                label="Budget Confirmed Probability Boost",
                description="Expected probability increase when budget is confirmed",
                type=ParameterType.PERCENTAGE,
                category=ParameterCategory.DEAL_LEVEL,
                default_value=25,
                min_value=10,
                max_value=40,
                depends_on_fields=['BudgetStatus', 'Probability'],
                business_context="Budget confirmation should significantly boost probability",
                data_source="crm",
                impact_level="high",
                expert_mode_only=True
            ),
            ParameterDefinition(
                key="timeline_confirmed_probability_floor",
                label="Timeline Confirmed Probability Floor",
                description="Minimum probability when implementation timeline is set",
                type=ParameterType.PERCENTAGE,
                category=ParameterCategory.DEAL_LEVEL,
                default_value=60,
                min_value=40,
                max_value=85,
                depends_on_fields=['TimelineStatus', 'Probability'],
                business_context="Timeline planning indicates serious buying intent",
                data_source="crm",
                impact_level="medium",
                expert_mode_only=True
            ),
            ParameterDefinition(
                key="technical_requirements_met_multiplier",
                label="Technical Requirements Met Multiplier",
                description="Risk reduction when technical requirements are satisfied",
                type=ParameterType.MULTIPLIER,
                category=ParameterCategory.DEAL_LEVEL,
                default_value=0.7,
                min_value=0.3,
                max_value=1.0,
                step=0.1,
                depends_on_fields=['TechnicalStatus'],
                business_context="Technical approval reduces sandbagging risk",
                data_source="crm",
                impact_level="medium",
                expert_mode_only=True
            ),
            ParameterDefinition(
                key="pricing_approved_probability_minimum",
                label="Pricing Approved Minimum Probability",
                description="Minimum probability when pricing is approved by customer",
                type=ParameterType.PERCENTAGE,
                category=ParameterCategory.DEAL_LEVEL,
                default_value=70,
                min_value=50,
                max_value=90,
                depends_on_fields=['PricingStatus', 'Probability'],
                business_context="Price acceptance indicates strong buying intent",
                data_source="crm",
                impact_level="high",
                expert_mode_only=True
            ),
            ParameterDefinition(
                key="legal_review_probability_threshold",
                label="Legal Review Probability Threshold",
                description="Expected probability when legal review has started",
                type=ParameterType.PERCENTAGE,
                category=ParameterCategory.DEAL_LEVEL,
                default_value=80,
                min_value=60,
                max_value=95,
                depends_on_fields=['LegalStatus', 'Probability'],
                business_context="Legal review indicates imminent closure",
                data_source="crm",
                impact_level="high",
                expert_mode_only=True
            ),
            ParameterDefinition(
                key="implementation_planning_multiplier",
                label="Implementation Planning Risk Multiplier",
                description="Risk reduction when implementation is being planned",
                type=ParameterType.MULTIPLIER,
                category=ParameterCategory.DEAL_LEVEL,
                default_value=0.6,
                min_value=0.3,
                max_value=1.0,
                step=0.1,
                depends_on_fields=['ImplementationStatus'],
                business_context="Implementation planning indicates deal commitment",
                data_source="crm",
                impact_level="medium",
                expert_mode_only=True
            ),
            ParameterDefinition(
                key="stakeholder_alignment_threshold",
                label="Stakeholder Alignment Threshold",
                description="Minimum stakeholder alignment score for probability validation",
                type=ParameterType.SCORE,
                category=ParameterCategory.DEAL_LEVEL,
                default_value=75,
                min_value=40,
                max_value=100,
                depends_on_fields=['StakeholderMap'],
                business_context="Poor stakeholder alignment with high probability is suspicious",
                data_source="calculated",
                calculation_method="stakeholder_alignment_score",
                impact_level="medium",
                expert_mode_only=True
            ),
            ParameterDefinition(
                key="urgency_vs_probability_correlation",
                label="Urgency vs Probability Correlation Factor",
                description="Expected correlation between customer urgency and deal probability",
                type=ParameterType.CORRELATION,
                category=ParameterCategory.DEAL_LEVEL,
                default_value=0.7,
                min_value=0.3,
                max_value=1.0,
                step=0.1,
                depends_on_fields=['UrgencyLevel', 'Probability'],
                business_context="High urgency should correlate with high probability",
                data_source="crm",
                impact_level="medium",
                expert_mode_only=True
            ),
            ParameterDefinition(
                key="deal_complexity_adjustment",
                label="Deal Complexity Adjustment Factor",
                description="Adjustment for deal complexity in sandbagging analysis",
                type=ParameterType.MULTIPLIER,
                category=ParameterCategory.DEAL_LEVEL,
                default_value=1.2,
                min_value=0.8,
                max_value=2.0,
                step=0.1,
                depends_on_fields=['ComplexityScore'],
                business_context="Complex deals may legitimately have lower probabilities",
                data_source="calculated",
                calculation_method="complexity_score_from_deal_attributes",
                impact_level="medium",
                expert_mode_only=True
            ),
            ParameterDefinition(
                key="customer_enthusiasm_threshold",
                label="Customer Enthusiasm Threshold",
                description="Minimum customer enthusiasm score for probability validation",
                type=ParameterType.SCORE,
                category=ParameterCategory.DEAL_LEVEL,
                default_value=70,
                min_value=30,
                max_value=100,
                depends_on_fields=['CustomerSentiment'],
                business_context="High enthusiasm with low probability indicates sandbagging",
                data_source="calculated",
                calculation_method="sentiment_analysis_from_communications",
                impact_level="high",
                expert_mode_only=True
            ),
            ParameterDefinition(
                key="champion_strength_multiplier",
                label="Internal Champion Strength Multiplier",
                description="Risk adjustment based on internal champion strength",
                type=ParameterType.MULTIPLIER,
                category=ParameterCategory.DEAL_LEVEL,
                default_value=0.8,
                min_value=0.5,
                max_value=1.0,
                step=0.1,
                depends_on_fields=['ChampionStrength'],
                business_context="Strong champions reduce sandbagging risk",
                data_source="calculated",
                calculation_method="champion_strength_from_engagement",
                impact_level="medium",
                expert_mode_only=True
            ),
            ParameterDefinition(
                key="deal_momentum_velocity_threshold",
                label="Deal Momentum Velocity Threshold",
                description="Minimum momentum velocity for active deals",
                type=ParameterType.VELOCITY,
                category=ParameterCategory.DEAL_LEVEL,
                default_value=2.5,
                min_value=1.0,
                max_value=10.0,
                step=0.5,
                depends_on_fields=['ActivityFrequency', 'StageProgression'],
                business_context="High momentum with low probability suggests sandbagging",
                data_source="calculated",
                calculation_method="momentum_from_activity_and_progression",
                impact_level="high",
                expert_mode_only=True
            ),
            ParameterDefinition(
                key="risk_assessment_probability_gap",
                label="Risk Assessment vs Probability Gap",
                description="Maximum acceptable gap between risk assessment and probability",
                type=ParameterType.PERCENTAGE,
                category=ParameterCategory.DEAL_LEVEL,
                default_value=30,
                min_value=10,
                max_value=60,
                depends_on_fields=['RiskAssessment', 'Probability'],
                business_context="Low risk with low probability may indicate sandbagging",
                data_source="crm",
                impact_level="medium",
                expert_mode_only=True
            ),
            ParameterDefinition(
                key="next_steps_clarity_threshold",
                label="Next Steps Clarity Threshold",
                description="Minimum clarity score for next steps definition",
                type=ParameterType.SCORE,
                category=ParameterCategory.DEAL_LEVEL,
                default_value=80,
                min_value=40,
                max_value=100,
                depends_on_fields=['NextSteps'],
                business_context="Vague next steps may indicate sandbagging or poor deal management",
                data_source="calculated",
                calculation_method="next_steps_clarity_nlp_analysis",
                impact_level="medium",
                expert_mode_only=True
            )
        ]
    
    def _get_rep_behavior_parameters(self) -> List[ParameterDefinition]:
        """Generate 25 rep behavior sandbagging parameters"""
        return [
            ParameterDefinition(
                key="rep_historical_sandbagging_score",
                label="Rep Historical Sandbagging Score",
                description="Rep's historical tendency to sandbag based on past performance",
                type=ParameterType.SCORE,
                category=ParameterCategory.REP_BEHAVIOR,
                default_value=50,
                min_value=0,
                max_value=100,
                depends_on_fields=['OwnerId'],
                business_context="Some reps consistently sandbag - factor in their history",
                data_source="historical",
                requires_historical_data=True,
                calculation_method="historical_forecast_vs_actual_variance",
                impact_level="critical"
            ),
            ParameterDefinition(
                key="forecast_vs_actual_variance_threshold",
                label="Forecast vs Actual Variance Threshold (%)",
                description="Acceptable variance between forecast and actual results",
                type=ParameterType.PERCENTAGE,
                category=ParameterCategory.REP_BEHAVIOR,
                default_value=20,
                min_value=5,
                max_value=50,
                depends_on_fields=['OwnerId'],
                business_context="Consistent under-forecasting indicates systematic sandbagging",
                data_source="historical",
                requires_historical_data=True,
                impact_level="critical"
            ),
            ParameterDefinition(
                key="deal_velocity_by_rep_threshold",
                label="Rep Deal Velocity Threshold",
                description="Minimum expected deal velocity for each rep",
                type=ParameterType.VELOCITY,
                category=ParameterCategory.REP_BEHAVIOR,
                default_value=1.5,
                min_value=0.5,
                max_value=5.0,
                step=0.1,
                depends_on_fields=['OwnerId'],
                business_context="Unusually slow deal progression may indicate sandbagging",
                data_source="calculated",
                calculation_method="rep_average_deal_velocity",
                impact_level="high",
                expert_mode_only=True
            ),
            ParameterDefinition(
                key="probability_update_frequency_threshold",
                label="Probability Update Frequency Threshold",
                description="Minimum frequency of probability updates per month",
                type=ParameterType.FREQUENCY,
                category=ParameterCategory.REP_BEHAVIOR,
                default_value=2,
                min_value=1,
                max_value=10,
                depends_on_fields=['OwnerId', 'Probability'],
                business_context="Infrequent probability updates may hide sandbagging",
                data_source="historical",
                requires_historical_data=True,
                impact_level="medium",
                expert_mode_only=True
            ),
            ParameterDefinition(
                key="last_minute_deal_addition_threshold",
                label="Last Minute Deal Addition Threshold",
                description="Maximum acceptable percentage of deals added in last week of quarter",
                type=ParameterType.PERCENTAGE,
                category=ParameterCategory.REP_BEHAVIOR,
                default_value=15,
                min_value=5,
                max_value=40,
                depends_on_fields=['OwnerId', 'CreatedDate'],
                business_context="Sudden appearance of deals suggests hidden pipeline",
                data_source="calculated",
                calculation_method="quarter_end_deal_creation_percentage",
                impact_level="high",
                expert_mode_only=True
            ),
            ParameterDefinition(
                key="quarter_end_surge_multiplier",
                label="Quarter End Surge Multiplier",
                description="Multiplier for deals closing in last 2 weeks of quarter",
                type=ParameterType.MULTIPLIER,
                category=ParameterCategory.REP_BEHAVIOR,
                default_value=1.8,
                min_value=1.0,
                max_value=3.0,
                step=0.1,
                depends_on_fields=['OwnerId', 'CloseDate'],
                business_context="Unnatural quarter-end spikes indicate sandbagging",
                data_source="calculated",
                calculation_method="quarter_end_closure_pattern",
                impact_level="high"
            ),
            ParameterDefinition(
                key="pipeline_coverage_inflation_threshold",
                label="Pipeline Coverage Inflation Threshold",
                description="Maximum acceptable pipeline coverage ratio",
                type=ParameterType.RATIO,
                category=ParameterCategory.REP_BEHAVIOR,
                default_value=4.0,
                min_value=2.0,
                max_value=10.0,
                step=0.5,
                depends_on_fields=['OwnerId'],
                business_context="Artificially inflated pipeline coverage may hide sandbagging",
                data_source="calculated",
                calculation_method="pipeline_to_quota_ratio",
                impact_level="medium",
                expert_mode_only=True
            ),
            ParameterDefinition(
                key="activity_logging_consistency_threshold",
                label="Activity Logging Consistency Threshold",
                description="Minimum consistency score for activity logging",
                type=ParameterType.SCORE,
                category=ParameterCategory.REP_BEHAVIOR,
                default_value=75,
                min_value=40,
                max_value=100,
                depends_on_fields=['OwnerId', 'ActivityDate'],
                business_context="Inconsistent logging may indicate manipulation",
                data_source="calculated",
                calculation_method="activity_logging_pattern_analysis",
                impact_level="medium",
                expert_mode_only=True
            ),
            ParameterDefinition(
                key="crm_data_quality_threshold",
                label="CRM Data Quality Threshold",
                description="Minimum data quality score per rep",
                type=ParameterType.SCORE,
                category=ParameterCategory.REP_BEHAVIOR,
                default_value=80,
                min_value=50,
                max_value=100,
                depends_on_fields=['OwnerId'],
                business_context="Poor data quality may indicate intentional obfuscation",
                data_source="calculated",
                calculation_method="data_completeness_and_accuracy_score",
                impact_level="medium"
            ),
            ParameterDefinition(
                key="forecast_submission_timeliness_threshold",
                label="Forecast Submission Timeliness Threshold (hours)",
                description="Maximum acceptable delay in forecast submission",
                type=ParameterType.COUNT,
                category=ParameterCategory.REP_BEHAVIOR,
                default_value=24,
                min_value=2,
                max_value=168,
                depends_on_fields=['OwnerId'],
                business_context="Late submissions may indicate last-minute manipulation",
                data_source="historical",
                requires_historical_data=True,
                impact_level="low",
                expert_mode_only=True
            ),
            ParameterDefinition(
                key="deal_notes_quality_threshold",
                label="Deal Notes Quality Threshold",
                description="Minimum quality score for deal progression notes",
                type=ParameterType.SCORE,
                category=ParameterCategory.REP_BEHAVIOR,
                default_value=70,
                min_value=30,
                max_value=100,
                depends_on_fields=['OwnerId', 'Description'],
                business_context="Vague notes may hide true deal status",
                data_source="calculated",
                calculation_method="notes_quality_nlp_analysis",
                impact_level="medium",
                expert_mode_only=True
            ),
            ParameterDefinition(
                key="stakeholder_mapping_completeness_threshold",
                label="Stakeholder Mapping Completeness Threshold",
                description="Minimum completeness score for stakeholder mapping",
                type=ParameterType.SCORE,
                category=ParameterCategory.REP_BEHAVIOR,
                default_value=80,
                min_value=40,
                max_value=100,
                depends_on_fields=['OwnerId', 'StakeholderMap'],
                business_context="Incomplete stakeholder info may hide deal risks",
                data_source="calculated",
                calculation_method="stakeholder_mapping_completeness",
                impact_level="medium",
                expert_mode_only=True
            ),
            ParameterDefinition(
                key="competitive_intelligence_depth_threshold",
                label="Competitive Intelligence Depth Threshold",
                description="Minimum depth score for competitive analysis",
                type=ParameterType.SCORE,
                category=ParameterCategory.REP_BEHAVIOR,
                default_value=60,
                min_value=20,
                max_value=100,
                depends_on_fields=['OwnerId', 'CompetitorAnalysis'],
                business_context="Lack of competitive intel may indicate poor deal qualification",
                data_source="calculated",
                calculation_method="competitive_analysis_depth_score",
                impact_level="medium",
                expert_mode_only=True
            ),
            ParameterDefinition(
                key="meeting_frequency_threshold",
                label="Meeting Frequency Threshold (per month)",
                description="Minimum meeting frequency for active deals",
                type=ParameterType.FREQUENCY,
                category=ParameterCategory.REP_BEHAVIOR,
                default_value=4,
                min_value=1,
                max_value=20,
                depends_on_fields=['OwnerId', 'ActivityType'],
                business_context="Low meeting frequency on high-probability deals is suspicious",
                data_source="calculated",
                calculation_method="meeting_frequency_from_activities",
                impact_level="medium"
            ),
            ParameterDefinition(
                key="email_engagement_rate_threshold",
                label="Email Engagement Rate Threshold (%)",
                description="Minimum email engagement rate for active deals",
                type=ParameterType.PERCENTAGE,
                category=ParameterCategory.REP_BEHAVIOR,
                default_value=40,
                min_value=10,
                max_value=80,
                depends_on_fields=['OwnerId', 'EmailEngagement'],
                business_context="Low email engagement with high probability suggests sandbagging",
                data_source="external",
                requires_external_data=True,
                impact_level="medium",
                expert_mode_only=True
            ),
            ParameterDefinition(
                key="phone_call_frequency_threshold",
                label="Phone Call Frequency Threshold (per month)",
                description="Minimum phone call frequency for high-value deals",
                type=ParameterType.FREQUENCY,
                category=ParameterCategory.REP_BEHAVIOR,
                default_value=6,
                min_value=2,
                max_value=20,
                depends_on_fields=['OwnerId', 'ActivityType'],
                business_context="Infrequent calls on valuable deals may indicate low confidence",
                data_source="calculated",
                calculation_method="call_frequency_from_activities",
                impact_level="medium",
                expert_mode_only=True
            ),
            ParameterDefinition(
                key="proposal_response_time_threshold",
                label="Proposal Response Time Threshold (days)",
                description="Maximum acceptable time to respond to RFPs",
                type=ParameterType.DAYS,
                category=ParameterCategory.REP_BEHAVIOR,
                default_value=5,
                min_value=1,
                max_value=30,
                depends_on_fields=['OwnerId', 'ProposalDate'],
                business_context="Slow RFP response may indicate low prioritization",
                data_source="calculated",
                calculation_method="rfp_response_time_analysis",
                impact_level="low",
                expert_mode_only=True
            ),
            ParameterDefinition(
                key="follow_up_consistency_score_threshold",
                label="Follow-up Consistency Score Threshold",
                description="Minimum consistency score for follow-up activities",
                type=ParameterType.SCORE,
                category=ParameterCategory.REP_BEHAVIOR,
                default_value=75,
                min_value=40,
                max_value=100,
                depends_on_fields=['OwnerId', 'ActivityDate'],
                business_context="Inconsistent follow-up may indicate poor deal management",
                data_source="calculated",
                calculation_method="follow_up_pattern_consistency",
                impact_level="medium",
                expert_mode_only=True
            ),
            ParameterDefinition(
                key="deal_review_participation_rate",
                label="Deal Review Participation Rate (%)",
                description="Percentage of deal reviews attended by rep",
                type=ParameterType.PERCENTAGE,
                category=ParameterCategory.REP_BEHAVIOR,
                default_value=90,
                min_value=50,
                max_value=100,
                depends_on_fields=['OwnerId'],
                business_context="Avoiding deal reviews may indicate sandbagging",
                data_source="historical",
                requires_historical_data=True,
                impact_level="medium",
                expert_mode_only=True
            ),
            ParameterDefinition(
                key="coaching_receptiveness_score",
                label="Coaching Receptiveness Score",
                description="Rep's receptiveness to pipeline coaching",
                type=ParameterType.SCORE,
                category=ParameterCategory.REP_BEHAVIOR,
                default_value=70,
                min_value=20,
                max_value=100,
                depends_on_fields=['OwnerId'],
                business_context="Resistance to coaching may indicate hidden information",
                data_source="external",
                requires_external_data=True,
                impact_level="low",
                expert_mode_only=True
            ),
            ParameterDefinition(
                key="transparency_score_threshold",
                label="Deal Transparency Score Threshold",
                description="Minimum transparency score for deal sharing",
                type=ParameterType.SCORE,
                category=ParameterCategory.REP_BEHAVIOR,
                default_value=80,
                min_value=40,
                max_value=100,
                depends_on_fields=['OwnerId'],
                business_context="Reluctance to share details may indicate sandbagging",
                data_source="calculated",
                calculation_method="transparency_from_data_completeness",
                impact_level="medium",
                expert_mode_only=True
            ),
            ParameterDefinition(
                key="cross_sell_identification_rate",
                label="Cross-sell Identification Rate (%)",
                description="Percentage of deals with identified expansion opportunities",
                type=ParameterType.PERCENTAGE,
                category=ParameterCategory.REP_BEHAVIOR,
                default_value=30,
                min_value=10,
                max_value=70,
                depends_on_fields=['OwnerId', 'ExpansionOpportunities'],
                business_context="Low expansion identification may indicate poor qualification",
                data_source="calculated",
                calculation_method="expansion_opportunity_identification_rate",
                impact_level="low",
                expert_mode_only=True
            ),
            ParameterDefinition(
                key="referral_generation_rate",
                label="Referral Generation Rate (%)",
                description="Percentage of deals generating referrals",
                type=ParameterType.PERCENTAGE,
                category=ParameterCategory.REP_BEHAVIOR,
                default_value=15,
                min_value=5,
                max_value=50,
                depends_on_fields=['OwnerId', 'ReferralSource'],
                business_context="Low referral rates may indicate poor customer relationships",
                data_source="calculated",
                calculation_method="referral_generation_rate",
                impact_level="low",
                expert_mode_only=True
            ),
            ParameterDefinition(
                key="win_loss_analysis_participation",
                label="Win/Loss Analysis Participation Rate (%)",
                description="Percentage of completed win/loss analyses by rep",
                type=ParameterType.PERCENTAGE,
                category=ParameterCategory.REP_BEHAVIOR,
                default_value=85,
                min_value=50,
                max_value=100,
                depends_on_fields=['OwnerId'],
                business_context="Avoiding loss analysis may indicate sandbagging patterns",
                data_source="historical",
                requires_historical_data=True,
                impact_level="medium",
                expert_mode_only=True
            ),
            ParameterDefinition(
                key="deal_qualification_depth_score",
                label="Deal Qualification Depth Score",
                description="Depth score for deal qualification (BANT, MEDDIC, etc.)",
                type=ParameterType.SCORE,
                category=ParameterCategory.REP_BEHAVIOR,
                default_value=75,
                min_value=40,
                max_value=100,
                depends_on_fields=['OwnerId', 'QualificationCriteria'],
                business_context="Poor qualification may lead to sandbagging",
                data_source="calculated",
                calculation_method="qualification_completeness_score",
                impact_level="high",
                expert_mode_only=True
            )
        ]
    
    def _get_temporal_parameters(self) -> List[ParameterDefinition]:
        """Generate 15 temporal pattern sandbagging parameters"""
        return [
            ParameterDefinition(
                key="seasonal_sandbagging_weight",
                label="Seasonal Sandbagging Weight",
                description="Weight adjustment for seasonal sandbagging patterns",
                type=ParameterType.WEIGHT,
                category=ParameterCategory.TEMPORAL_PATTERNS,
                default_value=0.3,
                min_value=0.0,
                max_value=1.0,
                step=0.05,
                depends_on_fields=['CloseDate'],
                business_context="Q4 often has more sandbagging due to quota pressure",
                data_source="calculated",
                calculation_method="seasonal_pattern_analysis",
                impact_level="high"
            ),
            ParameterDefinition(
                key="month_end_behavior_multiplier",
                label="Month End Behavior Multiplier",
                description="Risk multiplier for deals closing at month end",
                type=ParameterType.MULTIPLIER,
                category=ParameterCategory.TEMPORAL_PATTERNS,
                default_value=1.4,
                min_value=1.0,
                max_value=2.5,
                step=0.1,
                depends_on_fields=['CloseDate'],
                business_context="Unnatural month-end clustering may indicate manipulation",
                data_source="calculated",
                calculation_method="month_end_clustering_analysis",
                impact_level="medium"
            ),
            ParameterDefinition(
                key="quota_period_alignment_threshold",
                label="Quota Period Alignment Threshold",
                description="Maximum acceptable alignment with quota periods",
                type=ParameterType.PERCENTAGE,
                category=ParameterCategory.TEMPORAL_PATTERNS,
                default_value=40,
                min_value=20,
                max_value=70,
                depends_on_fields=['CloseDate'],
                business_context="Over-alignment with quota periods suggests gaming",
                data_source="calculated",
                calculation_method="quota_period_correlation",
                impact_level="high",
                expert_mode_only=True
            ),
            ParameterDefinition(
                key="holiday_period_activity_anomaly",
                label="Holiday Period Activity Anomaly Factor",
                description="Anomaly factor for unusual activity during holidays",
                type=ParameterType.MULTIPLIER,
                category=ParameterCategory.TEMPORAL_PATTERNS,
                default_value=2.0,
                min_value=1.0,
                max_value=5.0,
                step=0.2,
                depends_on_fields=['ActivityDate'],
                business_context="Unusual holiday activity may indicate deal manipulation",
                data_source="calculated",
                calculation_method="holiday_activity_pattern",
                impact_level="medium",
                expert_mode_only=True
            ),
            ParameterDefinition(
                key="fiscal_year_end_surge_threshold",
                label="Fiscal Year End Surge Threshold",
                description="Maximum acceptable deal surge at fiscal year end",
                type=ParameterType.PERCENTAGE,
                category=ParameterCategory.TEMPORAL_PATTERNS,
                default_value=25,
                min_value=10,
                max_value=50,
                depends_on_fields=['CloseDate'],
                business_context="Excessive year-end surges indicate sandbagging",
                data_source="calculated",
                calculation_method="fiscal_year_end_analysis",
                impact_level="high"
            ),
            ParameterDefinition(
                key="commission_period_correlation",
                label="Commission Period Correlation Factor",
                description="Correlation factor between deal timing and commission periods",
                type=ParameterType.CORRELATION,
                category=ParameterCategory.TEMPORAL_PATTERNS,
                default_value=0.6,
                min_value=0.2,
                max_value=1.0,
                step=0.1,
                depends_on_fields=['CloseDate'],
                business_context="High correlation with commission timing suggests gaming",
                data_source="calculated",
                calculation_method="commission_timing_correlation",
                impact_level="medium",
                expert_mode_only=True
            ),
            ParameterDefinition(
                key="performance_review_timing_weight",
                label="Performance Review Timing Weight",
                description="Weight for deals timed around performance reviews",
                type=ParameterType.WEIGHT,
                category=ParameterCategory.TEMPORAL_PATTERNS,
                default_value=0.4,
                min_value=0.1,
                max_value=0.8,
                step=0.1,
                depends_on_fields=['CloseDate'],
                business_context="Deal timing around reviews may indicate sandbagging",
                data_source="external",
                requires_external_data=True,
                impact_level="medium",
                expert_mode_only=True
            ),
            ParameterDefinition(
                key="territory_planning_period_impact",
                label="Territory Planning Period Impact Factor",
                description="Impact factor during territory planning periods",
                type=ParameterType.MULTIPLIER,
                category=ParameterCategory.TEMPORAL_PATTERNS,
                default_value=1.3,
                min_value=1.0,
                max_value=2.0,
                step=0.1,
                depends_on_fields=['CloseDate'],
                business_context="Territory changes may trigger sandbagging behavior",
                data_source="external",
                requires_external_data=True,
                impact_level="low",
                expert_mode_only=True
            ),
            ParameterDefinition(
                key="product_launch_timing_correlation",
                label="Product Launch Timing Correlation",
                description="Correlation between deal timing and product launches",
                type=ParameterType.CORRELATION,
                category=ParameterCategory.TEMPORAL_PATTERNS,
                default_value=0.5,
                min_value=0.2,
                max_value=1.0,
                step=0.1,
                depends_on_fields=['CloseDate', 'ProductType'],
                business_context="Deals timed with launches may be artificially delayed",
                data_source="external",
                requires_external_data=True,
                impact_level="low",
                expert_mode_only=True
            ),
            ParameterDefinition(
                key="customer_budget_cycle_alignment",
                label="Customer Budget Cycle Alignment Factor",
                description="Alignment factor with customer budget cycles",
                type=ParameterType.MULTIPLIER,
                category=ParameterCategory.TEMPORAL_PATTERNS,
                default_value=0.8,
                min_value=0.5,
                max_value=1.2,
                step=0.1,
                depends_on_fields=['CloseDate', 'CustomerBudgetCycle'],
                business_context="Budget cycle exploitation may indicate sandbagging",
                data_source="external",
                requires_external_data=True,
                impact_level="medium",
                expert_mode_only=True
            ),
            ParameterDefinition(
                key="industry_event_timing_weight",
                label="Industry Event Timing Weight",
                description="Weight for deals timed around industry events",
                type=ParameterType.WEIGHT,
                category=ParameterCategory.TEMPORAL_PATTERNS,
                default_value=0.2,
                min_value=0.0,
                max_value=0.6,
                step=0.1,
                depends_on_fields=['CloseDate', 'Industry'],
                business_context="Event timing may be used to justify deal delays",
                data_source="external",
                requires_external_data=True,
                impact_level="low",
                expert_mode_only=True
            ),
            ParameterDefinition(
                key="vacation_period_delay_factor",
                label="Vacation Period Delay Factor",
                description="Factor for deals delayed around vacation periods",
                type=ParameterType.MULTIPLIER,
                category=ParameterCategory.TEMPORAL_PATTERNS,
                default_value=1.2,
                min_value=1.0,
                max_value=1.8,
                step=0.1,
                depends_on_fields=['CloseDate'],
                business_context="Vacation-timed delays may indicate sandbagging",
                data_source="external",
                requires_external_data=True,
                impact_level="low",
                expert_mode_only=True
            ),
            ParameterDefinition(
                key="training_period_impact_multiplier",
                label="Training Period Impact Multiplier",
                description="Impact multiplier during sales training periods",
                type=ParameterType.MULTIPLIER,
                category=ParameterCategory.TEMPORAL_PATTERNS,
                default_value=1.1,
                min_value=1.0,
                max_value=1.5,
                step=0.1,
                depends_on_fields=['ActivityDate'],
                business_context="Training periods may affect deal progression patterns",
                data_source="external",
                requires_external_data=True,
                impact_level="low",
                expert_mode_only=True
            ),
            ParameterDefinition(
                key="leadership_transition_weight",
                label="Leadership Transition Weight",
                description="Weight factor during leadership transitions",
                type=ParameterType.WEIGHT,
                category=ParameterCategory.TEMPORAL_PATTERNS,
                default_value=0.5,
                min_value=0.2,
                max_value=0.8,
                step=0.1,
                depends_on_fields=['CloseDate'],
                business_context="Leadership changes may trigger conservative forecasting",
                data_source="external",
                requires_external_data=True,
                impact_level="medium",
                expert_mode_only=True
            ),
            ParameterDefinition(
                key="audit_period_behavior_multiplier",
                label="Audit Period Behavior Multiplier",
                description="Behavior change multiplier during audit periods",
                type=ParameterType.MULTIPLIER,
                category=ParameterCategory.TEMPORAL_PATTERNS,
                default_value=0.7,
                min_value=0.4,
                max_value=1.0,
                step=0.1,
                depends_on_fields=['CloseDate'],
                business_context="Audit periods may reduce sandbagging behavior",
                data_source="external",
                requires_external_data=True,
                impact_level="low",
                expert_mode_only=True
            )
        ]
    
    def _get_organizational_parameters(self) -> List[ParameterDefinition]:
        """Generate 15 organizational sandbagging parameters"""
        return [
            ParameterDefinition(
                key="quota_achievement_history_weight",
                label="Quota Achievement History Weight",
                description="Weight based on historical quota attainment patterns",
                type=ParameterType.WEIGHT,
                category=ParameterCategory.ORGANIZATIONAL,
                default_value=0.4,
                min_value=0.1,
                max_value=0.8,
                step=0.1,
                depends_on_fields=['OwnerId'],
                business_context="Historical quota performance affects sandbagging likelihood",
                data_source="historical",
                requires_historical_data=True,
                calculation_method="quota_attainment_pattern_analysis",
                impact_level="high"
            ),
            ParameterDefinition(
                key="compensation_plan_change_impact",
                label="Compensation Plan Change Impact Factor",
                description="Impact factor when compensation plans change",
                type=ParameterType.MULTIPLIER,
                category=ParameterCategory.ORGANIZATIONAL,
                default_value=1.6,
                min_value=1.0,
                max_value=2.5,
                step=0.1,
                depends_on_fields=['OwnerId'],
                business_context="Comp plan changes may trigger sandbagging behavior",
                data_source="external",
                requires_external_data=True,
                impact_level="high",
                expert_mode_only=True
            ),
            ParameterDefinition(
                key="territory_size_adjustment",
                label="Territory Size Adjustment Factor",
                description="Adjustment factor based on territory characteristics",
                type=ParameterType.MULTIPLIER,
                category=ParameterCategory.ORGANIZATIONAL,
                default_value=1.0,
                min_value=0.5,
                max_value=2.0,
                step=0.1,
                depends_on_fields=['Territory'],
                business_context="Territory size may affect sandbagging patterns",
                data_source="external",
                requires_external_data=True,
                impact_level="medium",
                expert_mode_only=True
            ),
            ParameterDefinition(
                key="team_performance_variance_threshold",
                label="Team Performance Variance Threshold",
                description="Maximum acceptable variance from team average",
                type=ParameterType.PERCENTAGE,
                category=ParameterCategory.ORGANIZATIONAL,
                default_value=30,
                min_value=10,
                max_value=60,
                depends_on_fields=['OwnerId'],
                business_context="Significant deviation from team norms may indicate sandbagging",
                data_source="calculated",
                calculation_method="team_performance_comparison",
                impact_level="medium"
            ),
            ParameterDefinition(
                key="management_attention_correlation",
                label="Management Attention Correlation Factor",
                description="Correlation between management oversight and performance",
                type=ParameterType.CORRELATION,
                category=ParameterCategory.ORGANIZATIONAL,
                default_value=0.7,
                min_value=0.3,
                max_value=1.0,
                step=0.1,
                depends_on_fields=['OwnerId'],
                business_context="High management attention may reduce sandbagging",
                data_source="external",
                requires_external_data=True,
                impact_level="medium",
                expert_mode_only=True
            ),
            ParameterDefinition(
                key="peer_performance_comparison_weight",
                label="Peer Performance Comparison Weight",
                description="Weight for peer performance comparison analysis",
                type=ParameterType.WEIGHT,
                category=ParameterCategory.ORGANIZATIONAL,
                default_value=0.3,
                min_value=0.1,
                max_value=0.7,
                step=0.1,
                depends_on_fields=['OwnerId'],
                business_context="Peer comparison reveals relative sandbagging patterns",
                data_source="calculated",
                calculation_method="peer_group_performance_analysis",
                impact_level="medium"
            ),
            ParameterDefinition(
                key="promotion_timing_impact",
                label="Promotion Timing Impact Factor",
                description="Impact factor around promotion opportunities",
                type=ParameterType.MULTIPLIER,
                category=ParameterCategory.ORGANIZATIONAL,
                default_value=1.4,
                min_value=1.0,
                max_value=2.0,
                step=0.1,
                depends_on_fields=['OwnerId'],
                business_context="Promotion timing may affect forecasting behavior",
                data_source="external",
                requires_external_data=True,
                impact_level="medium",
                expert_mode_only=True
            ),
            ParameterDefinition(
                key="training_program_effectiveness_score",
                label="Training Program Effectiveness Score",
                description="Effectiveness score of sales training programs",
                type=ParameterType.SCORE,
                category=ParameterCategory.ORGANIZATIONAL,
                default_value=75,
                min_value=30,
                max_value=100,
                depends_on_fields=['OwnerId'],
                business_context="Training effectiveness may reduce sandbagging",
                data_source="external",
                requires_external_data=True,
                impact_level="medium",
                expert_mode_only=True
            ),
            ParameterDefinition(
                key="crm_system_change_impact",
                label="CRM System Change Impact Factor",
                description="Impact factor during CRM system changes",
                type=ParameterType.MULTIPLIER,
                category=ParameterCategory.ORGANIZATIONAL,
                default_value=1.3,
                min_value=1.0,
                max_value=2.0,
                step=0.1,
                depends_on_fields=['LastModifiedDate'],
                business_context="System changes may affect data quality and behavior",
                data_source="external",
                requires_external_data=True,
                impact_level="low",
                expert_mode_only=True
            ),
            ParameterDefinition(
                key="process_change_adaptation_score",
                label="Process Change Adaptation Score",
                description="Score for adaptation to sales process changes",
                type=ParameterType.SCORE,
                category=ParameterCategory.ORGANIZATIONAL,
                default_value=70,
                min_value=30,
                max_value=100,
                depends_on_fields=['OwnerId'],
                business_context="Poor adaptation may increase sandbagging",
                data_source="external",
                requires_external_data=True,
                impact_level="low",
                expert_mode_only=True
            ),
            ParameterDefinition(
                key="incentive_program_participation_rate",
                label="Incentive Program Participation Rate (%)",
                description="Participation rate in company incentive programs",
                type=ParameterType.PERCENTAGE,
                category=ParameterCategory.ORGANIZATIONAL,
                default_value=80,
                min_value=40,
                max_value=100,
                depends_on_fields=['OwnerId'],
                business_context="Low participation may indicate disengagement",
                data_source="external",
                requires_external_data=True,
                impact_level="low",
                expert_mode_only=True
            ),
            ParameterDefinition(
                key="recognition_program_impact_score",
                label="Recognition Program Impact Score",
                description="Impact score of recognition programs on behavior",
                type=ParameterType.SCORE,
                category=ParameterCategory.ORGANIZATIONAL,
                default_value=60,
                min_value=20,
                max_value=100,
                depends_on_fields=['OwnerId'],
                business_context="Recognition may reduce sandbagging behavior",
                data_source="external",
                requires_external_data=True,
                impact_level="low",
                expert_mode_only=True
            ),
            ParameterDefinition(
                key="internal_communication_quality_score",
                label="Internal Communication Quality Score",
                description="Quality score of internal communication systems",
                type=ParameterType.SCORE,
                category=ParameterCategory.ORGANIZATIONAL,
                default_value=75,
                min_value=30,
                max_value=100,
                depends_on_fields=[],
                business_context="Poor communication may increase sandbagging",
                data_source="external",
                requires_external_data=True,
                impact_level="medium",
                expert_mode_only=True
            ),
            ParameterDefinition(
                key="cross_department_collaboration_score",
                label="Cross-Department Collaboration Score",
                description="Score for collaboration between sales and other departments",
                type=ParameterType.SCORE,
                category=ParameterCategory.ORGANIZATIONAL,
                default_value=70,
                min_value=30,
                max_value=100,
                depends_on_fields=['OwnerId'],
                business_context="Poor collaboration may affect deal progression",
                data_source="external",
                requires_external_data=True,
                impact_level="medium",
                expert_mode_only=True
            ),
            ParameterDefinition(
                key="company_culture_alignment_score",
                label="Company Culture Alignment Score",
                description="Score for individual alignment with company culture",
                type=ParameterType.SCORE,
                category=ParameterCategory.ORGANIZATIONAL,
                default_value=80,
                min_value=40,
                max_value=100,
                depends_on_fields=['OwnerId'],
                business_context="Cultural misalignment may increase sandbagging risk",
                data_source="external",
                requires_external_data=True,
                impact_level="medium",
                expert_mode_only=True
            )
        ]
    
    def _get_market_customer_parameters(self) -> List[ParameterDefinition]:
        """Generate 20 market & customer sandbagging parameters"""
        return [
            ParameterDefinition(
                key="industry_benchmark_adjustment",
                label="Industry Benchmark Adjustment Factor",
                description="Adjustment factor based on industry-specific patterns",
                type=ParameterType.MULTIPLIER,
                category=ParameterCategory.MARKET_CUSTOMER,
                default_value=1.0,
                min_value=0.5,
                max_value=2.0,
                step=0.1,
                depends_on_fields=['Industry'],
                business_context="Different industries have different probability patterns",
                data_source="external",
                requires_external_data=True,
                impact_level="high"
            ),
            ParameterDefinition(
                key="customer_size_segmentation_weight",
                label="Customer Size Segmentation Weight",
                description="Weight adjustment based on customer size segment",
                type=ParameterType.WEIGHT,
                category=ParameterCategory.MARKET_CUSTOMER,
                default_value=0.4,
                min_value=0.1,
                max_value=0.8,
                step=0.1,
                depends_on_fields=['AccountSize'],
                business_context="Enterprise vs SMB deals have different sandbagging patterns",
                data_source="crm",
                impact_level="medium"
            ),
            ParameterDefinition(
                key="geographic_market_adjustment",
                label="Geographic Market Adjustment Factor",
                description="Adjustment factor for different geographic markets",
                type=ParameterType.MULTIPLIER,
                category=ParameterCategory.MARKET_CUSTOMER,
                default_value=1.0,
                min_value=0.6,
                max_value=1.8,
                step=0.1,
                depends_on_fields=['BillingCountry'],
                business_context="Regional variations in sandbagging patterns",
                data_source="external",
                requires_external_data=True,
                impact_level="medium",
                expert_mode_only=True
            ),
            ParameterDefinition(
                key="economic_indicator_correlation",
                label="Economic Indicator Correlation Factor",
                description="Correlation factor with economic conditions",
                type=ParameterType.CORRELATION,
                category=ParameterCategory.MARKET_CUSTOMER,
                default_value=0.5,
                min_value=0.2,
                max_value=1.0,
                step=0.1,
                depends_on_fields=['CloseDate'],
                business_context="Economic conditions may affect sandbagging behavior",
                data_source="external",
                requires_external_data=True,
                impact_level="medium",
                expert_mode_only=True
            ),
            ParameterDefinition(
                key="competitive_landscape_intensity",
                label="Competitive Landscape Intensity Factor",
                description="Intensity factor based on competitive environment",
                type=ParameterType.MULTIPLIER,
                category=ParameterCategory.MARKET_CUSTOMER,
                default_value=1.2,
                min_value=0.8,
                max_value=2.0,
                step=0.1,
                depends_on_fields=['Industry', 'CompetitorAnalysis'],
                business_context="High competition may increase sandbagging",
                data_source="external",
                requires_external_data=True,
                impact_level="high",
                expert_mode_only=True
            ),
            ParameterDefinition(
                key="customer_buying_behavior_score",
                label="Customer Buying Behavior Score",
                description="Score based on customer buying behavior patterns",
                type=ParameterType.SCORE,
                category=ParameterCategory.MARKET_CUSTOMER,
                default_value=70,
                min_value=30,
                max_value=100,
                depends_on_fields=['AccountId'],
                business_context="Customer behavior affects deal probability accuracy",
                data_source="calculated",
                calculation_method="customer_buying_pattern_analysis",
                impact_level="medium",
                expert_mode_only=True
            ),
            ParameterDefinition(
                key="market_maturity_adjustment",
                label="Market Maturity Adjustment Factor",
                description="Adjustment factor based on market maturity level",
                type=ParameterType.MULTIPLIER,
                category=ParameterCategory.MARKET_CUSTOMER,
                default_value=1.0,
                min_value=0.7,
                max_value=1.5,
                step=0.1,
                depends_on_fields=['Industry'],
                business_context="Market maturity affects deal predictability",
                data_source="external",
                requires_external_data=True,
                impact_level="medium",
                expert_mode_only=True
            ),
            ParameterDefinition(
                key="regulatory_environment_impact",
                label="Regulatory Environment Impact Factor",
                description="Impact factor of regulatory environment on deals",
                type=ParameterType.MULTIPLIER,
                category=ParameterCategory.MARKET_CUSTOMER,
                default_value=1.1,
                min_value=0.8,
                max_value=1.8,
                step=0.1,
                depends_on_fields=['Industry', 'BillingCountry'],
                business_context="Regulatory complexity may affect deal timing",
                data_source="external",
                requires_external_data=True,
                impact_level="medium",
                expert_mode_only=True
            ),
            ParameterDefinition(
                key="technology_adoption_rate_factor",
                label="Technology Adoption Rate Factor",
                description="Factor based on customer technology adoption rate",
                type=ParameterType.MULTIPLIER,
                category=ParameterCategory.MARKET_CUSTOMER,
                default_value=1.0,
                min_value=0.6,
                max_value=1.6,
                step=0.1,
                depends_on_fields=['Industry', 'ProductType'],
                business_context="Tech adoption rate affects deal velocity",
                data_source="external",
                requires_external_data=True,
                impact_level="low",
                expert_mode_only=True
            ),
            ParameterDefinition(
                key="customer_budget_cycle_correlation",
                label="Customer Budget Cycle Correlation",
                description="Correlation with customer budget cycles",
                type=ParameterType.CORRELATION,
                category=ParameterCategory.MARKET_CUSTOMER,
                default_value=0.6,
                min_value=0.2,
                max_value=1.0,
                step=0.1,
                depends_on_fields=['CloseDate', 'Industry'],
                business_context="Budget cycles may be exploited for sandbagging",
                data_source="external",
                requires_external_data=True,
                impact_level="medium",
                expert_mode_only=True
            ),
            ParameterDefinition(
                key="decision_making_authority_score",
                label="Decision Making Authority Score",
                description="Score based on customer decision-making authority",
                type=ParameterType.SCORE,
                category=ParameterCategory.MARKET_CUSTOMER,
                default_value=75,
                min_value=30,
                max_value=100,
                depends_on_fields=['ContactRole'],
                business_context="Decision authority affects deal progression predictability",
                data_source="calculated",
                calculation_method="decision_authority_analysis",
                impact_level="medium",
                expert_mode_only=True
            ),
            ParameterDefinition(
                key="procurement_process_complexity_factor",
                label="Procurement Process Complexity Factor",
                description="Factor based on customer procurement process complexity",
                type=ParameterType.MULTIPLIER,
                category=ParameterCategory.MARKET_CUSTOMER,
                default_value=1.3,
                min_value=1.0,
                max_value=2.2,
                step=0.1,
                depends_on_fields=['AccountSize', 'Industry'],
                business_context="Complex procurement may legitimately delay deals",
                data_source="external",
                requires_external_data=True,
                impact_level="medium",
                expert_mode_only=True
            ),
            ParameterDefinition(
                key="customer_risk_tolerance_score",
                label="Customer Risk Tolerance Score",
                description="Score based on customer risk tolerance level",
                type=ParameterType.SCORE,
                category=ParameterCategory.MARKET_CUSTOMER,
                default_value=60,
                min_value=20,
                max_value=100,
                depends_on_fields=['Industry', 'AccountSize'],
                business_context="Risk tolerance affects deal probability accuracy",
                data_source="external",
                requires_external_data=True,
                impact_level="medium",
                expert_mode_only=True
            ),
            ParameterDefinition(
                key="innovation_adoption_willingness",
                label="Innovation Adoption Willingness Score",
                description="Score for customer willingness to adopt innovations",
                type=ParameterType.SCORE,
                category=ParameterCategory.MARKET_CUSTOMER,
                default_value=65,
                min_value=20,
                max_value=100,
                depends_on_fields=['Industry', 'ProductType'],
                business_context="Innovation willingness affects new product deal probability",
                data_source="external",
                requires_external_data=True,
                impact_level="low",
                expert_mode_only=True
            ),
            ParameterDefinition(
                key="partnership_ecosystem_influence",
                label="Partnership Ecosystem Influence Factor",
                description="Influence factor of partner ecosystem on deals",
                type=ParameterType.MULTIPLIER,
                category=ParameterCategory.MARKET_CUSTOMER,
                default_value=1.1,
                min_value=0.8,
                max_value=1.5,
                step=0.1,
                depends_on_fields=['PartnerInfluence'],
                business_context="Partner influence may affect deal timing and probability",
                data_source="external",
                requires_external_data=True,
                impact_level="low",
                expert_mode_only=True
            ),
            ParameterDefinition(
                key="market_demand_fluctuation_factor",
                label="Market Demand Fluctuation Factor",
                description="Factor based on market demand variability",
                type=ParameterType.MULTIPLIER,
                category=ParameterCategory.MARKET_CUSTOMER,
                default_value=1.0,
                min_value=0.6,
                max_value=1.8,
                step=0.1,
                depends_on_fields=['Industry', 'ProductType'],
                business_context="Demand fluctuations may affect deal predictability",
                data_source="external",
                requires_external_data=True,
                impact_level="medium",
                expert_mode_only=True
            ),
            ParameterDefinition(
                key="customer_retention_pattern_score",
                label="Customer Retention Pattern Score",
                description="Score based on existing customer retention patterns",
                type=ParameterType.SCORE,
                category=ParameterCategory.MARKET_CUSTOMER,
                default_value=80,
                min_value=40,
                max_value=100,
                depends_on_fields=['AccountId'],
                business_context="Retention patterns affect expansion deal probability",
                data_source="calculated",
                calculation_method="customer_retention_analysis",
                impact_level="medium",
                expert_mode_only=True
            ),
            ParameterDefinition(
                key="new_vs_existing_customer_factor",
                label="New vs Existing Customer Factor",
                description="Factor differentiating new vs existing customer deals",
                type=ParameterType.MULTIPLIER,
                category=ParameterCategory.MARKET_CUSTOMER,
                default_value=1.4,
                min_value=1.0,
                max_value=2.0,
                step=0.1,
                depends_on_fields=['AccountId'],
                business_context="New customer deals have higher uncertainty",
                data_source="calculated",
                calculation_method="customer_relationship_age",
                impact_level="medium"
            ),
            ParameterDefinition(
                key="customer_success_health_score",
                label="Customer Success Health Score",
                description="Health score from customer success metrics",
                type=ParameterType.SCORE,
                category=ParameterCategory.MARKET_CUSTOMER,
                default_value=75,
                min_value=30,
                max_value=100,
                depends_on_fields=['AccountId'],
                business_context="Customer health affects expansion deal probability",
                data_source="external",
                requires_external_data=True,
                impact_level="high",
                expert_mode_only=True
            ),
            ParameterDefinition(
                key="market_share_dynamics_impact",
                label="Market Share Dynamics Impact Factor",
                description="Impact factor based on competitive market position",
                type=ParameterType.MULTIPLIER,
                category=ParameterCategory.MARKET_CUSTOMER,
                default_value=1.0,
                min_value=0.7,
                max_value=1.5,
                step=0.1,
                depends_on_fields=['Industry'],
                business_context="Market position affects deal win probability",
                data_source="external",
                requires_external_data=True,
                impact_level="medium",
                expert_mode_only=True
            )
        ]

# Global instance
parameter_discovery = DynamicParameterDiscovery()
