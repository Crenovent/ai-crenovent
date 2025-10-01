"""
Business Rules Engine
Configuration-driven rule execution for RBA workflows

This engine replaces hardcoded business logic with YAML-defined rules,
enabling business users to modify rules without code changes.
"""

import logging
import yaml
import os
from typing import Dict, List, Any, Optional, Union
from datetime import datetime, timedelta
from dataclasses import dataclass
import re

logger = logging.getLogger(__name__)

@dataclass
class RuleEvaluationResult:
    """Result of rule evaluation"""
    rule_name: str
    score: float
    weight: float
    weighted_score: float
    condition_met: bool
    details: Dict[str, Any]
    confidence_factors: List[str]

@dataclass
class ScoringResult:
    """Complete scoring result from rules engine"""
    total_score: float
    risk_level: str
    confidence: float
    rule_results: List[RuleEvaluationResult]
    recommendations: List[str]
    metadata: Dict[str, Any]

class BusinessRulesEngine:
    """
    Configuration-driven business rules engine for RBA workflows
    
    Features:
    - YAML-defined business rules
    - Dynamic threshold evaluation
    - Industry-specific adjustments
    - Configurable scoring algorithms
    - Template-based rule inheritance
    """
    
    def __init__(self, rules_config_path: str = None):
        self.logger = logging.getLogger(__name__)
        
        # Set default rules configuration path
        if rules_config_path is None:
            current_dir = os.path.dirname(os.path.abspath(__file__))
            self.rules_config_path = os.path.join(current_dir, "business_rules.yaml")
        else:
            self.rules_config_path = rules_config_path
            
        # Load rules configuration
        self.rules = {}
        self.templates = {}
        self._load_rules_config()
    
    def _load_rules_config(self) -> bool:
        """Load business rules from YAML configuration"""
        try:
            if not os.path.exists(self.rules_config_path):
                self.logger.error(f"Rules configuration file not found: {self.rules_config_path}")
                return False
                
            with open(self.rules_config_path, 'r') as file:
                config = yaml.safe_load(file)
                
            self.rules = config.get('rules', {})
            self.templates = config.get('templates', {})
            
            self.logger.info(f"✅ Loaded {len(self.rules)} rule sets and {len(self.templates)} templates")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to load rules configuration: {e}")
            return False
    
    def evaluate_rules(
        self,
        rule_set_name: str,
        opportunity: Dict[str, Any], 
        config: Dict[str, Any],
        industry: str = None
    ) -> ScoringResult:
        """
        DYNAMIC RULE EVALUATION - Works with any rule set
        
        This method can evaluate any workflow rules defined in the YAML configuration.
        No hardcoding required - it dynamically discovers and executes rules.
        """
        
        if rule_set_name not in self.rules:
            raise ValueError(f"Rule set '{rule_set_name}' not found in configuration")
            
        rule_set = self.rules[rule_set_name]
        
        # Apply industry-specific adjustments
        if industry:
            rule_set = self._apply_industry_adjustments(rule_set, industry)
        
        # Merge runtime config with rule defaults
        effective_config = self._merge_config(rule_set.get('defaults', {}), config)
        
        # Evaluate all scoring factors
        rule_results = []
        total_score = 0.0
        confidence_factors = []
        
        for factor in rule_set.get('scoring_factors', []):
            result = self._evaluate_scoring_factor(factor, opportunity, effective_config)
            rule_results.append(result)
            
            if result.condition_met:
                total_score += result.weighted_score
                confidence_factors.extend(result.confidence_factors)
        
        # Calculate risk level and confidence
        risk_level = self._calculate_risk_level(total_score, rule_set.get('risk_thresholds', {}))
        confidence = min(100.0, (len(confidence_factors) / 4.0) * 100.0)
        
        # Generate recommendations
        recommendations = self._generate_recommendations(rule_results, risk_level, rule_set)
        
        return ScoringResult(
            total_score=total_score,
            risk_level=risk_level,
            confidence=confidence,
            rule_results=rule_results,
            recommendations=recommendations,
            metadata={
                'rule_set': rule_set_name,
                'industry': industry,
                'evaluation_timestamp': datetime.now().isoformat(),
                'config_used': effective_config
            }
        )
    
    def evaluate_sandbagging_rules(
        self, 
        opportunity: Dict[str, Any], 
        config: Dict[str, Any],
        industry: str = None
    ) -> ScoringResult:
        """Backward compatibility method for sandbagging rules"""
        return self.evaluate_rules("sandbagging_detection", opportunity, config, industry)
    
    def evaluate_pipeline_hygiene_rules(
        self,
        opportunities: List[Dict[str, Any]],
        config: Dict[str, Any],
        analysis_type: str = "stale_deals"
    ) -> Dict[str, Any]:
        """Evaluate pipeline hygiene rules"""
        
        rule_set_name = f"pipeline_hygiene_{analysis_type}"
        if rule_set_name not in self.rules:
            rule_set_name = "pipeline_hygiene_default"
            
        if rule_set_name not in self.rules:
            raise ValueError(f"Pipeline hygiene rule set not found: {rule_set_name}")
            
        rule_set = self.rules[rule_set_name]
        effective_config = self._merge_config(rule_set.get('defaults', {}), config)
        
        # Process all opportunities
        flagged_opportunities = []
        total_opportunities = len(opportunities)
        
        for opp in opportunities:
            # Evaluate each opportunity against the rules
            if self._evaluate_opportunity_conditions(opp, rule_set.get('conditions', []), effective_config):
                flagged_opp = self._enrich_opportunity_data(opp, rule_set, effective_config)
                flagged_opportunities.append(flagged_opp)
        
        # Calculate overall metrics
        compliance_score = self._calculate_compliance_score(
            total_opportunities, 
            len(flagged_opportunities), 
            rule_set.get('compliance_threshold', 70)
        )
        
        return {
            'analysis_type': analysis_type,
            'total_opportunities': total_opportunities,
            'flagged_opportunities': len(flagged_opportunities),
            'flagged_deals': flagged_opportunities,
            'compliance_score': compliance_score,
            'compliance_status': "COMPLIANT" if compliance_score >= rule_set.get('compliance_threshold', 70) else "NON_COMPLIANT",
            'rule_set_used': rule_set_name,
            'evaluation_timestamp': datetime.now().isoformat()
        }
    
    def _evaluate_scoring_factor(
        self, 
        factor: Dict[str, Any], 
        opportunity: Dict[str, Any], 
        config: Dict[str, Any]
    ) -> RuleEvaluationResult:
        """Evaluate a single scoring factor"""
        
        factor_name = factor.get('name', 'unnamed_factor')
        weight = factor.get('weight', 0)
        multiplier = self._resolve_config_value(factor.get('multiplier', 1.0), config)
        
        # Evaluate conditions
        conditions = factor.get('condition', {})
        condition_met = self._evaluate_conditions(conditions, opportunity, config)
        
        # Calculate score
        score = 0.0
        confidence_factors = []
        
        if condition_met:
            score = weight * multiplier
            confidence_factors.append(self._generate_confidence_message(factor, opportunity, config))
        
        return RuleEvaluationResult(
            rule_name=factor_name,
            score=score,
            weight=weight,
            weighted_score=score,
            condition_met=condition_met,
            details={
                'conditions': conditions,
                'multiplier': multiplier,
                'opportunity_data': self._extract_relevant_data(opportunity, conditions)
            },
            confidence_factors=confidence_factors
        )
    
    def _evaluate_conditions(
        self, 
        conditions: Dict[str, Any], 
        opportunity: Dict[str, Any], 
        config: Dict[str, Any]
    ) -> bool:
        """Evaluate rule conditions against opportunity data"""
        
        for field, condition_value in conditions.items():
            opp_value = opportunity.get(field)
            
            if opp_value is None:
                continue
                
            # Handle different condition types
            if isinstance(condition_value, str):
                if condition_value.startswith('>='):
                    threshold = self._resolve_config_value(condition_value[2:].strip(), config)
                    if not (float(opp_value) >= float(threshold)):
                        return False
                elif condition_value.startswith('<='):
                    threshold = self._resolve_config_value(condition_value[2:].strip(), config)
                    if not (float(opp_value) <= float(threshold)):
                        return False
                elif condition_value.startswith('>'):
                    threshold = self._resolve_config_value(condition_value[1:].strip(), config)
                    if not (float(opp_value) > float(threshold)):
                        return False
                elif condition_value.startswith('<'):
                    threshold = self._resolve_config_value(condition_value[1:].strip(), config)
                    if not (float(opp_value) < float(threshold)):
                        return False
                elif '-' in condition_value:
                    # Range condition like "0-30"
                    range_parts = condition_value.split('-')
                    if len(range_parts) == 2:
                        min_val = float(range_parts[0])
                        max_val = float(range_parts[1])
                        if not (min_val <= float(opp_value) <= max_val):
                            return False
            
            elif isinstance(condition_value, list):
                # Check if any keywords match (for stage_keywords, etc.)
                opp_str = str(opp_value).lower()
                if not any(keyword.lower() in opp_str for keyword in condition_value):
                    return False
            
            elif isinstance(condition_value, (int, float)):
                if float(opp_value) != float(condition_value):
                    return False
            
            elif isinstance(condition_value, bool) and condition_value:
                # Handle special boolean conditions dynamically
                evaluator = self._get_condition_evaluator(field)
                if evaluator:
                    if not evaluator(opportunity, config):
                        return False
                else:
                    # Unknown boolean condition, log warning and assume it passes
                    self.logger.warning(f"Unknown boolean condition: {field}")
        
        return True
    
    def _resolve_config_value(self, value: Union[str, int, float], config: Dict[str, Any]) -> Union[str, int, float]:
        """Resolve configuration variables like ${high_value_threshold}"""
        
        if isinstance(value, str) and value.startswith('${') and value.endswith('}'):
            var_name = value[2:-1]
            return config.get(var_name, value)
        
        return value
    
    def _apply_industry_adjustments(self, rule_set: Dict[str, Any], industry: str) -> Dict[str, Any]:
        """Apply industry-specific adjustments to rule set"""
        
        adjustments = rule_set.get('industry_adjustments', {})
        if industry not in adjustments:
            return rule_set
            
        # Create a copy and apply adjustments
        adjusted_rule_set = rule_set.copy()
        industry_factor = adjustments[industry]
        
        # Apply multiplier to scoring factors
        if 'scoring_factors' in adjusted_rule_set:
            for factor in adjusted_rule_set['scoring_factors']:
                if 'weight' in factor:
                    factor['weight'] = int(factor['weight'] * industry_factor)
        
        return adjusted_rule_set
    
    def _merge_config(self, defaults: Dict[str, Any], runtime_config: Dict[str, Any]) -> Dict[str, Any]:
        """Merge default configuration with runtime configuration"""
        merged = defaults.copy()
        merged.update(runtime_config)
        return merged
    
    def _calculate_risk_level(self, score: float, thresholds: Dict[str, float]) -> str:
        """Calculate risk level based on score and thresholds"""
        
        if score >= thresholds.get('CRITICAL', 85):
            return 'CRITICAL'
        elif score >= thresholds.get('HIGH', 70):
            return 'HIGH'
        elif score >= thresholds.get('MEDIUM', 50):
            return 'MEDIUM'
        else:
            return 'LOW'
    
    def _generate_recommendations(
        self, 
        rule_results: List[RuleEvaluationResult], 
        risk_level: str, 
        rule_set: Dict[str, Any]
    ) -> List[str]:
        """Generate recommendations based on rule evaluation results"""
        
        recommendations = []
        rule_recommendations = rule_set.get('recommendations', {})
        
        # Add risk-level specific recommendations
        if risk_level in rule_recommendations:
            recommendations.extend(rule_recommendations[risk_level])
        
        # Add rule-specific recommendations
        for result in rule_results:
            if result.condition_met and result.rule_name in rule_recommendations:
                recommendations.extend(rule_recommendations[result.rule_name])
        
        return list(set(recommendations))  # Remove duplicates
    
    def _evaluate_opportunity_conditions(
        self, 
        opportunity: Dict[str, Any], 
        conditions: List[Dict[str, Any]], 
        config: Dict[str, Any]
    ) -> bool:
        """Evaluate if opportunity meets any of the specified conditions"""
        
        for condition_set in conditions:
            if self._evaluate_conditions(condition_set, opportunity, config):
                return True
        
        return False
    
    def _enrich_opportunity_data(
        self, 
        opportunity: Dict[str, Any], 
        rule_set: Dict[str, Any], 
        config: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Enrich opportunity data with rule-based analysis"""
        
        enriched = opportunity.copy()
        
        # Add rule-based fields
        enriched.update({
            'rule_set_used': rule_set.get('name', 'unknown'),
            'analysis_timestamp': datetime.now().isoformat(),
            'config_applied': config
        })
        
        return enriched
    
    def _calculate_compliance_score(
        self, 
        total_opportunities: int, 
        flagged_opportunities: int, 
        threshold: float
    ) -> float:
        """Calculate compliance score"""
        
        if total_opportunities == 0:
            return 100.0
            
        clean_opportunities = total_opportunities - flagged_opportunities
        return (clean_opportunities / total_opportunities) * 100.0
    
    def _generate_confidence_message(
        self, 
        factor: Dict[str, Any], 
        opportunity: Dict[str, Any], 
        config: Dict[str, Any]
    ) -> str:
        """Generate human-readable confidence message for a factor"""
        
        factor_name = factor.get('name', 'Unknown factor')
        
        # Extract relevant values for the message
        relevant_values = []
        for field, condition in factor.get('condition', {}).items():
            value = opportunity.get(field)
            if value is not None:
                relevant_values.append(f"{field}: {value}")
        
        return f"{factor_name} ({', '.join(relevant_values)})"
    
    def _extract_relevant_data(self, opportunity: Dict[str, Any], conditions: Dict[str, Any]) -> Dict[str, Any]:
        """Extract relevant opportunity data based on conditions"""
        
        relevant_data = {}
        for field in conditions.keys():
            if field in opportunity:
                relevant_data[field] = opportunity[field]
        
        return relevant_data
    
    def reload_rules(self) -> bool:
        """Reload rules configuration from file"""
        return self._load_rules_config()
    
    # ========================================
    # DYNAMIC CONDITION EVALUATORS
    # ========================================
    
    def _get_condition_evaluator(self, condition_name: str):
        """Get the appropriate evaluator function for a condition"""
        evaluators = {
            'stage_probability_check': self._evaluate_stage_probability_check,
            'velocity_benchmark_check': self._evaluate_velocity_benchmark_check,
            'coverage_ratio_check': self._evaluate_coverage_ratio_check,
            'stage_distribution_check': self._evaluate_stage_distribution_check,
            'critical_field_check': self._evaluate_critical_field_check,
            'important_field_check': self._evaluate_important_field_check,
            'stage_regression_check': self._evaluate_stage_regression_check,
            'activity_stagnation_check': self._evaluate_activity_stagnation_check,
            'high_value_stagnation_check': self._evaluate_high_value_stagnation_check,
            'duplicate_check': self._evaluate_duplicate_check,
            'data_quality_check': self._evaluate_data_quality_check,
        }
        return evaluators.get(condition_name)
    
    def _evaluate_stage_probability_check(self, opportunity: Dict[str, Any], config: Dict[str, Any]) -> bool:
        """Check if probability is too low for current stage"""
        stage_name = opportunity.get('StageName', '').strip()
        probability = opportunity.get('Probability', 0)
        
        # Get stage probability minimums from config
        stage_minimums = config.get('stage_probability_minimums', {})
        
        # Find matching stage (case-insensitive, partial match)
        required_probability = None
        for stage_key, min_prob in stage_minimums.items():
            if stage_key.lower() in stage_name.lower() or stage_name.lower() in stage_key.lower():
                required_probability = min_prob
                break
        
        if required_probability is None:
            return False  # Unknown stage, assume it passes
        
        # Return True if probability is below the minimum for this stage
        return probability < required_probability
    
    def _evaluate_velocity_benchmark_check(self, opportunity: Dict[str, Any], config: Dict[str, Any]) -> bool:
        """Check if deal velocity exceeds benchmarks"""
        stage_name = opportunity.get('StageName', '').strip().lower()
        days_in_stage = opportunity.get('DaysSinceCreated', 0)
        
        # Get velocity benchmarks from config
        velocity_benchmarks = config.get('velocity_benchmarks', {})
        stage_velocity_benchmarks = config.get('stage_velocity_benchmarks', {})
        
        # Combine both benchmark sources
        all_benchmarks = {**velocity_benchmarks, **stage_velocity_benchmarks}
        
        # Find matching stage benchmark
        benchmark_days = None
        for stage_key, max_days in all_benchmarks.items():
            if stage_key.lower() in stage_name or stage_name in stage_key.lower():
                benchmark_days = max_days
                break
        
        if benchmark_days is None:
            return False  # No benchmark, assume it passes
        
        # Check if deal exceeds benchmark
        slow_multiplier = config.get('slow_velocity_multiplier', 1.5)
        return days_in_stage > (benchmark_days * slow_multiplier)
    
    def _evaluate_coverage_ratio_check(self, opportunity: Dict[str, Any], config: Dict[str, Any]) -> bool:
        """Check pipeline coverage ratios - placeholder for now"""
        # This would need aggregated pipeline data, not individual opportunity data
        return False
    
    def _evaluate_stage_distribution_check(self, opportunity: Dict[str, Any], config: Dict[str, Any]) -> bool:
        """Check stage distribution - placeholder for now"""
        # This would need aggregated pipeline data, not individual opportunity data
        return False
    
    def _evaluate_critical_field_check(self, opportunity: Dict[str, Any], config: Dict[str, Any]) -> bool:
        """Check if critical fields are missing"""
        critical_fields = config.get('critical_fields', ['Amount', 'CloseDate', 'StageName', 'Probability'])
        
        for field in critical_fields:
            value = opportunity.get(field)
            if value is None or (isinstance(value, str) and value.strip() == '') or value == 0:
                return True  # Missing critical field
        
        return False  # All critical fields present
    
    def _evaluate_important_field_check(self, opportunity: Dict[str, Any], config: Dict[str, Any]) -> bool:
        """Check if important fields are missing"""
        important_fields = config.get('important_fields', ['LeadSource', 'Type', 'Owner.Name', 'Account.Name'])
        
        missing_count = 0
        for field in important_fields:
            if '.' in field:
                # Nested field like 'Owner.Name'
                parts = field.split('.')
                value = opportunity
                for part in parts:
                    value = value.get(part, {}) if isinstance(value, dict) else None
                    if value is None:
                        break
            else:
                value = opportunity.get(field)
            
            if value is None or (isinstance(value, str) and value.strip() == ''):
                missing_count += 1
        
        # Return True if more than half of important fields are missing
        return missing_count > len(important_fields) / 2
    
    def _evaluate_stage_regression_check(self, opportunity: Dict[str, Any], config: Dict[str, Any]) -> bool:
        """Check if deal moved backwards in stages"""
        # Would need historical stage data - placeholder for now
        return False
    
    def _evaluate_activity_stagnation_check(self, opportunity: Dict[str, Any], config: Dict[str, Any]) -> bool:
        """Check if deal has stagnant activity"""
        days_since_activity = opportunity.get('DaysSinceActivity', 0)
        minimum_activity_days = config.get('minimum_activity_days', 14)
        
        return days_since_activity > minimum_activity_days
    
    def _evaluate_high_value_stagnation_check(self, opportunity: Dict[str, Any], config: Dict[str, Any]) -> bool:
        """Check if high-value deal has stagnant activity"""
        amount = opportunity.get('Amount', 0)
        days_since_activity = opportunity.get('DaysSinceActivity', 0)
        
        high_value_threshold = config.get('high_value_threshold', 100000)
        activity_threshold = config.get('high_value_activity_threshold', 7)
        
        return amount >= high_value_threshold and days_since_activity > activity_threshold
    
    def _evaluate_duplicate_check(self, opportunity: Dict[str, Any], config: Dict[str, Any]) -> bool:
        """Check for potential duplicates - placeholder"""
        # Would need comparison with other opportunities - placeholder for now
        return False
    
    def _evaluate_data_quality_check(self, opportunity: Dict[str, Any], config: Dict[str, Any]) -> bool:
        """General data quality check"""
        data_quality_threshold = config.get('data_quality_threshold', 80)
        
        # Count total fields and populated fields
        total_fields = len(opportunity)
        populated_fields = sum(1 for value in opportunity.values() 
                             if value is not None and value != '' and value != 0)
        
        if total_fields == 0:
            return True  # No data is poor quality
        
        quality_percentage = (populated_fields / total_fields) * 100
        return quality_percentage < data_quality_threshold
