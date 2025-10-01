"""
RBA Configuration API
Endpoints for managing RBA analysis configurations
"""

from fastapi import APIRouter, HTTPException
from typing import Dict, Any
import logging

logger = logging.getLogger(__name__)

# Create RBA config router
router = APIRouter(prefix="/api/pipeline/config", tags=["RBA Configuration"])


@router.get("/{analysis_type}")
async def get_analysis_config(analysis_type: str):
    """Get configurable parameters for a specific analysis type"""
    try:
        from dsl.rules.business_rules_engine import BusinessRulesEngine
        
        # Load business rules to get default configs
        rules_engine = BusinessRulesEngine()
        
        if analysis_type not in rules_engine.rules:
            raise HTTPException(status_code=404, detail=f"Analysis type '{analysis_type}' not found")
        
        rule_set = rules_engine.rules[analysis_type]
        
        # Extract configurable parameters
        config_schema = {
            "analysis_type": analysis_type,
            "name": rule_set.get("name", analysis_type.replace('_', ' ').title()),
            "description": rule_set.get("description", ""),
            "user_configurable": rule_set.get("user_configurable", False),
            "defaults": rule_set.get("defaults", {}),
            "parameter_descriptions": _get_parameter_descriptions(analysis_type),
            "recommendations": rule_set.get("recommendations", {}),
            "available_rules": list(rules_engine.rules.keys())
        }
        
        return {
            "success": True,
            "config_schema": config_schema
        }
        
    except Exception as e:
        logger.error(f"❌ Failed to get config for {analysis_type}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get configuration: {str(e)}")


@router.post("/{analysis_type}")
async def update_analysis_config(analysis_type: str, config: Dict[str, Any]):
    """Update configuration for a specific analysis type"""
    try:
        # In a real system, this would save to database
        # For now, we'll just validate and return the config
        
        logger.info(f"🔧 Updating config for {analysis_type}: {config}")
        
        # Validate config parameters
        validation_result = _validate_config_parameters(analysis_type, config)
        
        if not validation_result["valid"]:
            raise HTTPException(status_code=400, detail=f"Invalid configuration: {validation_result['errors']}")
        
        return {
            "success": True,
            "message": f"Configuration updated for {analysis_type}",
            "updated_config": config,
            "validation": validation_result
        }
        
    except Exception as e:
        logger.error(f"❌ Failed to update config for {analysis_type}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to update configuration: {str(e)}")


@router.get("/")
async def get_all_configs():
    """Get all available RBA configurations"""
    try:
        from dsl.rules.business_rules_engine import BusinessRulesEngine
        
        rules_engine = BusinessRulesEngine()
        
        all_configs = {}
        for analysis_type in rules_engine.rules.keys():
            rule_set = rules_engine.rules[analysis_type]
            all_configs[analysis_type] = {
                "name": rule_set.get("name", analysis_type.replace('_', ' ').title()),
                "description": rule_set.get("description", ""),
                "user_configurable": rule_set.get("user_configurable", False),
                "defaults": rule_set.get("defaults", {}),
                "parameter_count": len(rule_set.get("defaults", {}))
            }
        
        return {
            "success": True,
            "total_configs": len(all_configs),
            "configurations": all_configs
        }
        
    except Exception as e:
        logger.error(f"❌ Failed to get all configs: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get configurations: {str(e)}")


def _get_parameter_descriptions(analysis_type: str) -> Dict[str, str]:
    """Get human-readable descriptions for configuration parameters"""
    
    descriptions = {
        "sandbagging_detection": {
            "high_value_threshold": "Minimum deal amount ($) to be considered high-value (default: $250,000)",
            "low_probability_threshold": "Probability (%) below which deals are flagged as suspicious (default: 35%)",
            "sandbagging_threshold": "Overall score threshold above which deals are flagged (default: 65)",
            "confidence_threshold": "Minimum confidence level required for flagging (default: 70%)",
            "stage_probability_minimums": "Minimum probability expected for each sales stage"
        },
        "stale_deals": {
            "stale_threshold_days": "Days without progression before deal is considered stale (default: 45)",
            "critical_threshold_days": "Days without progression before deal is critical (default: 75)",
            "minimum_activity_days": "Maximum days allowed without activity (default: 14)",
            "stage_velocity_benchmarks": "Expected days per sales stage"
        },
        "missing_fields": {
            "data_quality_threshold": "Minimum data completeness percentage required (default: 80%)",
            "minimum_amount_threshold": "Minimum deal amount requiring complete data (default: $1,000)",
            "critical_fields": "Fields that must be populated for all deals",
            "important_fields": "Fields that should be populated for quality"
        },
        "stage_velocity": {
            "slow_velocity_multiplier": "Multiplier for benchmark to identify slow deals (default: 1.5x)",
            "critical_velocity_multiplier": "Multiplier for benchmark to identify critical deals (default: 2.0x)",
            "velocity_benchmarks": "Expected velocity (days) for each sales stage"
        },
        "coverage_analysis": {
            "minimum_coverage_ratio": "Minimum pipeline to quota ratio (default: 3.0)",
            "healthy_coverage_ratio": "Healthy pipeline to quota ratio (default: 4.0)",
            "optimal_coverage_ratio": "Optimal pipeline to quota ratio (default: 5.0)",
            "quarterly_quota": "Quarterly sales quota in dollars (default: $1,000,000)",
            "pipeline_stages_included": "Sales stages to include in pipeline coverage calculation"
        }
    }
    
    return descriptions.get(analysis_type, {})


def _validate_config_parameters(analysis_type: str, config: Dict[str, Any]) -> Dict[str, Any]:
    """Validate configuration parameters for an analysis type"""
    
    errors = []
    warnings = []
    
    try:
        if analysis_type == "sandbagging_detection":
            if "high_value_threshold" in config:
                if not isinstance(config["high_value_threshold"], (int, float)) or config["high_value_threshold"] <= 0:
                    errors.append("high_value_threshold must be a positive number")
                elif config["high_value_threshold"] < 10000:
                    warnings.append("high_value_threshold below $10K may generate too many alerts")
            
            if "low_probability_threshold" in config:
                if not isinstance(config["low_probability_threshold"], (int, float)) or not (0 <= config["low_probability_threshold"] <= 100):
                    errors.append("low_probability_threshold must be between 0 and 100")
                    
            if "sandbagging_threshold" in config:
                if not isinstance(config["sandbagging_threshold"], (int, float)) or not (0 <= config["sandbagging_threshold"] <= 100):
                    errors.append("sandbagging_threshold must be between 0 and 100")
        
        elif analysis_type == "stale_deals":
            if "stale_threshold_days" in config:
                if not isinstance(config["stale_threshold_days"], int) or config["stale_threshold_days"] <= 0:
                    errors.append("stale_threshold_days must be a positive integer")
                elif config["stale_threshold_days"] < 7:
                    warnings.append("stale_threshold_days below 7 days may be too aggressive")
                    
            if "critical_threshold_days" in config:
                if not isinstance(config["critical_threshold_days"], int) or config["critical_threshold_days"] <= 0:
                    errors.append("critical_threshold_days must be a positive integer")
                elif config["critical_threshold_days"] <= config.get("stale_threshold_days", 30):
                    errors.append("critical_threshold_days must be greater than stale_threshold_days")
        
        elif analysis_type == "coverage_analysis":
            if "quarterly_quota" in config:
                if not isinstance(config["quarterly_quota"], (int, float)) or config["quarterly_quota"] <= 0:
                    errors.append("quarterly_quota must be a positive number")
                elif config["quarterly_quota"] < 50000:
                    warnings.append("quarterly_quota below $50K seems low for enterprise sales")
                    
            for ratio_field in ["minimum_coverage_ratio", "healthy_coverage_ratio", "optimal_coverage_ratio"]:
                if ratio_field in config:
                    if not isinstance(config[ratio_field], (int, float)) or config[ratio_field] <= 0:
                        errors.append(f"{ratio_field} must be a positive number")
        
        # Add more validation rules as needed
        
        return {
            "valid": len(errors) == 0,
            "errors": errors,
            "warnings": warnings
        }
        
    except Exception as e:
        return {
            "valid": False,
            "errors": [f"Validation error: {str(e)}"],
            "warnings": []
        }
