#!/usr/bin/env python3
"""
Clean Parameter Mapper - Dynamic Configuration
==============================================

Maps extracted parameters to workflow-specific configuration.
No hardcoded mappings - all dynamic based on workflow requirements.
"""

import logging
import re
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)

class CleanParameterMapper:
    """
    Clean Parameter Mapper
    
    Maps extracted parameters to workflow-specific configuration dynamically.
    Supports all time units (years, months, weeks, days) and various thresholds.
    """
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    def map_parameters(
        self, 
        parameters: Dict[str, Any], 
        workflow_category: str
    ) -> Dict[str, Any]:
        """
        Map extracted parameters to precise workflow configuration
        
        Args:
            parameters: Extracted parameters from deep LLM analysis
            workflow_category: Target workflow category
            
        Returns:
            Precise configuration for accurate workflow execution
        """
        try:
            self.logger.info(f"🧠 Deep mapping parameters for workflow: {workflow_category}")
            self.logger.info(f"📊 Business requirement: {parameters.get('business_requirement', 'Not specified')}")
            self.logger.info(f"🎯 Exact conditions: {parameters.get('exact_conditions', 'Not specified')}")
            
            # Base configuration
            mapped_config = {
                "data_source": "csv_data",
                "priority": "HIGH",  # All user requests are high priority
                "business_requirement": parameters.get("business_requirement", ""),
                "exact_conditions": parameters.get("exact_conditions", "")
            }
            
            # Workflow-specific precise mapping
            if workflow_category == "activity_tracking_audit":
                mapped_config.update(self._map_activity_audit_precise(parameters))
            elif workflow_category == "missing_fields_audit":
                mapped_config.update(self._map_missing_fields_precise(parameters))
            elif workflow_category == "duplicate_detection":
                mapped_config.update(self._map_duplicate_detection_precise(parameters))
            elif workflow_category == "ownerless_deals_detection":
                mapped_config.update(self._map_ownerless_deals_precise(parameters))
            elif workflow_category == "pipeline_hygiene_stale_deals":
                mapped_config.update(self._map_stale_deals_precise(parameters))
            elif workflow_category == "risk_scoring_analysis":
                mapped_config.update(self._map_risk_analysis_precise(parameters))
            elif workflow_category == "sandbagging_detection":
                mapped_config.update(self._map_sandbagging_detection_precise(parameters))
            elif workflow_category == "stage_velocity_analysis":
                mapped_config.update(self._map_stage_velocity_precise(parameters))
            elif workflow_category == "pipeline_health_overview":
                mapped_config.update(self._map_pipeline_health_precise(parameters))
            else:
                # Generic precise mapping
                mapped_config.update(self._map_generic_precise(parameters))
            
            # Add user-specific configuration
            if self._is_user_specific(parameters):
                mapped_config.update({
                    "scope": "user_specific",
                    "owner_filter": "current_user"
                })
            
            self.logger.info(f"✅ Precise configuration mapped: {mapped_config}")
            return mapped_config
            
        except Exception as e:
            self.logger.error(f"❌ Parameter mapping failed: {e}")
            return {"data_source": "csv_data", "priority": "HIGH", "error": str(e)}
    
    def _map_activity_audit_precise(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Map parameters for activity tracking audit with precise conditions"""
        config = {}
        
        # Extract activity threshold - "last 14 days" means check for activities in last 14 days
        activity_days = parameters.get("activity_threshold_days", 14)
        if "time_period_days" in parameters:
            activity_days = parameters["time_period_days"]
        
        config.update({
            "activity_threshold_days": activity_days,
            "missing_activities": parameters.get("missing_activities", True),
            "check_calls": True,
            "check_emails": True,
            "check_meetings": True,
            "filter_condition": f"no activity in last {activity_days} days",
            "sql_condition": f"last_activity_date IS NULL OR last_activity_date < (CURRENT_DATE - INTERVAL '{activity_days} days')"
        })
        
        return config
    
    def _map_missing_fields_precise(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Map parameters for missing fields audit with precise field requirements"""
        config = {}
        
        # Extract required fields from parameters
        required_fields = parameters.get("required_fields", [])
        if not required_fields:
            # Default based on common business requirements
            required_fields = ["close_date", "amount", "owner"]
        
        config.update({
            "required_fields": required_fields,
            "data_quality_check": parameters.get("data_quality_check", True),
            "completeness_threshold": 1.0,  # 100% - we want exact matches
            "filter_condition": f"missing {', '.join(required_fields)}",
            "sql_condition": " OR ".join([f"{field} IS NULL" for field in required_fields])
        })
        
        return config
    
    def _map_duplicate_detection_precise(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Map parameters for duplicate detection with precise matching"""
        config = {}
        
        config.update({
            "duplicate_check": parameters.get("duplicate_check", True),
            "match_fields": ["name", "account_id", "amount"],
            "similarity_threshold": 0.85,
            "fuzzy_matching": True,
            "filter_condition": "duplicate deals across CRM",
            "exact_match_required": False
        })
        
        return config
    
    def _map_ownerless_deals_precise(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Map parameters for ownerless deals detection with precise ownership check"""
        config = {}
        
        config.update({
            "owner_filter": parameters.get("owner_filter", "null"),
            "unassigned_check": True,
            "filter_condition": "ownerless or unassigned deals",
            "sql_condition": "owner IS NULL OR owner = '' OR owner = 'Unassigned'",
            "include_blank_owners": True
        })
        
        return config
    
    def _map_stale_deals_precise(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Map parameters for stale deals with precise time conditions"""
        config = {}
        
        # Extract time period - support years, months, days
        stale_days = parameters.get("time_period_days", 60)
        if "activity_threshold_days" in parameters:
            stale_days = parameters["activity_threshold_days"]
        
        config.update({
            "stale_threshold_days": stale_days,
            "critical_threshold_days": stale_days + 30,
            "filter_condition": f"deals stuck for more than {stale_days} days",
            "sql_condition": f"stage_duration_days > {stale_days}",
            "minimum_hygiene_score": 70
        })
        
        return config
    
    def _map_risk_analysis_precise(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Map parameters for risk analysis with precise risk thresholds"""
        config = {}
        
        threshold = parameters.get("threshold", "medium")
        if "high" in str(threshold).lower():
            risk_threshold = 0.8
        elif "low" in str(threshold).lower():
            risk_threshold = 0.3
        else:
            risk_threshold = 0.6
        
        config.update({
            "risk_threshold": risk_threshold,
            "categorize_risk": True,
            "filter_condition": f"risk scoring and categorization",
            "risk_levels": ["high", "medium", "low"]
        })
        
        return config
    
    def _map_generic_precise(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Generic precise mapping for any workflow"""
        config = {}
        
        # Apply time mapping if present
        if "time_period_days" in parameters:
            config["threshold_days"] = parameters["time_period_days"]
        if "activity_threshold_days" in parameters:
            config["activity_threshold_days"] = parameters["activity_threshold_days"]
        
        # Apply condition mapping
        if "exact_conditions" in parameters:
            config["filter_condition"] = parameters["exact_conditions"]
        if "business_requirement" in parameters:
            config["business_requirement"] = parameters["business_requirement"]
        
        return config
    
    def _map_stale_deals_parameters(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Map parameters for stale deals workflow"""
        config = {}
        
        # Time period mapping - support all units
        if "time_period_days" in parameters:
            stale_days = parameters["time_period_days"]
        elif "time_period" in parameters:
            stale_days = self._convert_time_to_days(parameters["time_period"])
        else:
            stale_days = 60  # Default
        
        config["stale_threshold_days"] = stale_days
        config["critical_threshold_days"] = stale_days + 30  # Add buffer
        config["minimum_hygiene_score"] = 70
        
        # Priority based on time period
        if stale_days >= 1000:  # ~3 years
            config["priority"] = "LOW"
        elif stale_days >= 180:  # ~6 months
            config["priority"] = "MEDIUM"
        else:
            config["priority"] = "HIGH"
        
        self.logger.info(f"🕐 Applied time mapping: {stale_days} days for pipeline_hygiene_stale_deals")
        return config
    
    def _map_activity_parameters(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Map parameters for activity tracking workflow"""
        config = {}
        
        # Activity threshold (default 14 days for recent activity)
        if "time_period_days" in parameters:
            activity_days = parameters["time_period_days"]
        else:
            activity_days = 14  # Default recent activity threshold
        
        config["activity_threshold_days"] = activity_days
        config["minimum_activity_score"] = 50
        
        return config
    
    def _map_risk_parameters(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Map parameters for risk scoring workflow"""
        config = {}
        
        # Risk thresholds
        threshold = parameters.get("threshold", "medium")
        
        if "high" in str(threshold).lower():
            config["risk_threshold"] = 0.8
            config["priority"] = "HIGH"
        elif "low" in str(threshold).lower():
            config["risk_threshold"] = 0.3
            config["priority"] = "LOW"
        else:  # medium or default
            config["risk_threshold"] = 0.6
            config["priority"] = "MEDIUM"
        
        config["minimum_risk_score"] = 0.1
        return config
    
    def _map_missing_fields_parameters(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Map parameters for missing fields audit workflow"""
        config = {}
        
        # Extract required fields from parameters
        required_fields = []
        raw_input = parameters.get("raw_input", "").lower()
        
        if "close date" in raw_input or "close_date" in raw_input:
            required_fields.append("close_date")
        if "amount" in raw_input:
            required_fields.append("amount")
        if "owner" in raw_input:
            required_fields.append("owner")
        if "activity" in raw_input:
            required_fields.append("last_activity_date")
        
        if not required_fields:
            required_fields = ["close_date", "amount", "owner"]  # Default critical fields
        
        config["required_fields"] = required_fields
        config["completeness_threshold"] = 0.9
        
        return config
    
    def _map_duplicate_parameters(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Map parameters for duplicate detection workflow"""
        config = {}
        
        # Duplicate matching criteria
        config["match_fields"] = ["name", "account_id", "amount"]
        config["similarity_threshold"] = 0.85
        config["fuzzy_matching"] = True
        
        return config
    
    def _map_generic_parameters(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Generic parameter mapping for unknown workflows"""
        config = {}
        
        # Apply time mapping if present
        if "time_period_days" in parameters:
            config["threshold_days"] = parameters["time_period_days"]
        elif "time_period" in parameters:
            config["threshold_days"] = self._convert_time_to_days(parameters["time_period"])
        
        # Apply entity mapping
        if "entities" in parameters:
            config["target_entities"] = parameters["entities"]
        
        # Apply action mapping
        if "actions" in parameters:
            config["required_actions"] = parameters["actions"]
        
        return config
    
    def _convert_time_to_days(self, time_period: str) -> int:
        """Convert time period string to days"""
        if not time_period:
            return 60  # Default
        
        time_lower = time_period.lower()
        
        # Extract number and unit
        patterns = [
            (r'(\d+)\s*years?', lambda m: int(m.group(1)) * 365),
            (r'(\d+)\s*months?', lambda m: int(m.group(1)) * 30),
            (r'(\d+)\s*weeks?', lambda m: int(m.group(1)) * 7),
            (r'(\d+)\s*days?', lambda m: int(m.group(1))),
            (r'>(\d+)\s*years?', lambda m: int(m.group(1)) * 365),
            (r'>(\d+)\s*months?', lambda m: int(m.group(1)) * 30),
            (r'>(\d+)\s*weeks?', lambda m: int(m.group(1)) * 7),
            (r'>(\d+)\s*days?', lambda m: int(m.group(1)))
        ]
        
        for pattern, converter in patterns:
            match = re.search(pattern, time_lower)
            if match:
                days = converter(match)
                self.logger.info(f"🕐 Converted '{time_period}' to {days} days")
                return days
        
        # If no pattern matches, try to extract just the number
        number_match = re.search(r'(\d+)', time_period)
        if number_match:
            days = int(number_match.group(1))
            self.logger.info(f"🕐 Extracted {days} days from '{time_period}'")
            return days
        
        return 60  # Default fallback
    
    def _is_user_specific(self, parameters: Dict[str, Any]) -> bool:
        """Check if request is user-specific (my deals, my opportunities)"""
        entities = parameters.get("entities", [])
        raw_input = parameters.get("raw_input", "").lower()
        
        return (
            any("my" in str(entity).lower() for entity in entities) or
            "my deals" in raw_input or
            "my opportunities" in raw_input
        )
    
    def _map_sandbagging_detection_precise(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Map parameters for sandbagging detection with user-provided thresholds"""
        config = {}
        
        # Extract threshold from user input
        threshold = parameters.get('threshold', '')
        percentage_threshold = parameters.get('percentage_threshold', '')
        amount_threshold = parameters.get('amount_threshold', '')
        
        # Apply user-provided thresholds with smart defaults
        if threshold:
            threshold_value = self._parse_percentage(str(threshold))
            if threshold_value:
                config['sandbagging_threshold'] = threshold_value
                config['low_probability_threshold'] = min(threshold_value, 30)  # Ensure logical relationship
        
        if percentage_threshold:
            perc_value = self._parse_percentage(str(percentage_threshold))
            if perc_value:
                config['sandbagging_threshold'] = perc_value
                config['low_probability_threshold'] = min(perc_value, 30)
        
        if amount_threshold:
            amount_value = self._parse_amount(str(amount_threshold))
            if amount_value:
                config['high_value_threshold'] = amount_value
        
        # Set defaults for missing values
        if 'sandbagging_threshold' not in config:
            config['sandbagging_threshold'] = 70
        if 'high_value_threshold' not in config:
            config['high_value_threshold'] = 100000
        if 'low_probability_threshold' not in config:
            config['low_probability_threshold'] = 30
        
        self.logger.info(f"🎯 Sandbagging config: threshold={config.get('sandbagging_threshold')}%, high_value=${config.get('high_value_threshold'):,}, low_prob={config.get('low_probability_threshold')}%")
        return config
    
    def _map_stage_velocity_precise(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Map parameters for stage velocity analysis with user-provided thresholds"""
        config = {}
        
        # Extract time period and thresholds
        time_period = parameters.get('time_period', '')
        threshold = parameters.get('threshold', '')
        multiplier_threshold = parameters.get('multiplier_threshold', '')
        
        # Apply user-provided values with smart defaults
        if time_period:
            days = self._parse_time_period(time_period)
            if days:
                config['velocity_threshold_days'] = days
                config['critical_velocity_days'] = days * 2  # Critical is 2x velocity threshold
        
        if threshold:
            threshold_value = self._parse_number(str(threshold))
            if threshold_value:
                config['velocity_threshold_days'] = threshold_value
                config['critical_velocity_days'] = threshold_value * 2
        
        if multiplier_threshold:
            multiplier = self._parse_multiplier(str(multiplier_threshold))
            if multiplier:
                config['bottleneck_threshold'] = multiplier
        
        # Set defaults for missing values
        if 'velocity_threshold_days' not in config:
            config['velocity_threshold_days'] = 30
        if 'critical_velocity_days' not in config:
            config['critical_velocity_days'] = 60
        if 'bottleneck_threshold' not in config:
            config['bottleneck_threshold'] = 1.5
        
        self.logger.info(f"🎯 Velocity config: velocity={config.get('velocity_threshold_days')} days, critical={config.get('critical_velocity_days')} days, bottleneck={config.get('bottleneck_threshold')}x")
        return config
    
    def _map_pipeline_health_precise(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Map parameters for pipeline health overview with user-provided thresholds"""
        config = {}
        
        # Extract coverage ratio and thresholds
        coverage_ratio = parameters.get('coverage_ratio', '')
        amount_threshold = parameters.get('amount_threshold', '')
        percentage_threshold = parameters.get('percentage_threshold', '')
        threshold = parameters.get('threshold', '')
        
        # Apply user-provided coverage ratio
        if coverage_ratio:
            ratio_value = self._parse_coverage_ratio(str(coverage_ratio))
            if ratio_value:
                config['coverage_ratio_target'] = ratio_value
        
        # Apply user-provided amount thresholds
        if amount_threshold:
            amount_value = self._parse_amount(str(amount_threshold))
            if amount_value:
                # Determine if it's large or medium deal threshold based on value
                if amount_value > 75000:  # Assume > $75K is large deal threshold
                    config['large_deal_threshold'] = amount_value
                else:
                    config['medium_deal_threshold'] = amount_value
        
        if threshold:
            threshold_value = self._parse_number(str(threshold))
            if threshold_value:
                if threshold_value > 10000:  # Amount threshold
                    config['large_deal_threshold'] = threshold_value
                elif threshold_value < 100:  # Percentage threshold
                    if threshold_value > 50:
                        config['high_probability_threshold'] = threshold_value
                    else:
                        config['low_probability_threshold'] = threshold_value
        
        if percentage_threshold:
            perc_value = self._parse_percentage(str(percentage_threshold))
            if perc_value:
                if perc_value > 50:
                    config['high_probability_threshold'] = perc_value
                else:
                    config['low_probability_threshold'] = perc_value
        
        # Set defaults for missing values
        if 'coverage_ratio_target' not in config:
            config['coverage_ratio_target'] = 3.0
        if 'quota_threshold' not in config:
            config['quota_threshold'] = 1000000
        if 'large_deal_threshold' not in config:
            config['large_deal_threshold'] = 100000
        if 'medium_deal_threshold' not in config:
            config['medium_deal_threshold'] = 50000
        if 'high_probability_threshold' not in config:
            config['high_probability_threshold'] = 80
        if 'low_probability_threshold' not in config:
            config['low_probability_threshold'] = 20
        if 'bottleneck_deal_threshold' not in config:
            config['bottleneck_deal_threshold'] = 200
        if 'low_prob_deal_threshold' not in config:
            config['low_prob_deal_threshold'] = 50
        
        self.logger.info(f"🎯 Health config: coverage={config.get('coverage_ratio_target')}x, large_deal=${config.get('large_deal_threshold'):,}, high_prob={config.get('high_probability_threshold')}%")
        return config
    
    def _parse_percentage(self, percentage_str: str) -> Optional[float]:
        """Parse percentage string to numeric value"""
        if not percentage_str:
            return None
        percentage_str = percentage_str.strip().replace('%', '')
        try:
            return float(percentage_str)
        except ValueError:
            return None
    
    def _parse_amount(self, amount_str: str) -> Optional[float]:
        """Parse amount string to numeric value"""
        if not amount_str:
            return None
        amount_str = amount_str.lower().strip().replace(',', '').replace('$', '')
        
        # Handle K, M, B suffixes
        if 'k' in amount_str:
            number = re.search(r'(\d+(?:\.\d+)?)', amount_str)
            if number:
                return float(number.group(1)) * 1000
        elif 'm' in amount_str:
            number = re.search(r'(\d+(?:\.\d+)?)', amount_str)
            if number:
                return float(number.group(1)) * 1000000
        elif 'b' in amount_str:
            number = re.search(r'(\d+(?:\.\d+)?)', amount_str)
            if number:
                return float(number.group(1)) * 1000000000
        else:
            try:
                return float(amount_str)
            except ValueError:
                return None
        return None
    
    def _parse_multiplier(self, multiplier_str: str) -> Optional[float]:
        """Parse multiplier string to numeric value"""
        if not multiplier_str:
            return None
        multiplier_str = multiplier_str.lower().strip().replace('x', '')
        try:
            return float(multiplier_str)
        except ValueError:
            return None
    
    def _parse_coverage_ratio(self, coverage_str: str) -> Optional[float]:
        """Parse coverage ratio string to numeric value"""
        if not coverage_str:
            return None
        coverage_str = coverage_str.lower().strip()
        
        # Extract number before 'x' or 'coverage' or 'ratio'
        patterns = [r'(\d+(?:\.\d+)?)\s*x', r'(\d+(?:\.\d+)?)\s*coverage', r'(\d+(?:\.\d+)?)\s*ratio']
        for pattern in patterns:
            match = re.search(pattern, coverage_str)
            if match:
                try:
                    return float(match.group(1))
                except ValueError:
                    continue
        return None
    
    def _parse_number(self, number_str: str) -> Optional[float]:
        """Parse any numeric string"""
        if not number_str:
            return None
        try:
            return float(number_str)
        except ValueError:
            return None

# Global instance for easy access
_parameter_mapper = None

def get_parameter_mapper() -> CleanParameterMapper:
    """Get global parameter mapper instance"""
    global _parameter_mapper
    if _parameter_mapper is None:
        _parameter_mapper = CleanParameterMapper()
    return _parameter_mapper
