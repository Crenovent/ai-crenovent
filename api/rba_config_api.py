"""
RBA Configuration API
=====================
REST API endpoints for managing RBA agent configurations.
Provides dynamic configuration management, validation, and UI schema generation.
"""

from fastapi import APIRouter, HTTPException, Query, Body
from typing import Dict, List, Any, Optional
import logging
from datetime import datetime

from dsl.operators.rba.sandbagging_rba_agent import SandbaggingRBAAgent
from dsl.operators.rba.pipeline_summary_rba_agent import PipelineSummaryRBAAgent

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/rba-config", tags=["RBA Configuration"])

# Dynamic agent registry - automatically discovers all agents
async def get_agent_registry():
    """Get all available RBA agents dynamically"""
    try:
        # Try the existing RBA registry first
        from dsl.registry.enhanced_capability_registry import EnhancedCapabilityRegistry
        rba_registry = EnhancedCapabilityRegistry()
        await rba_registry.initialize()
        
        agents = {}
        for agent_name, agent_info in rba_registry.get_all_agents().items():
            agents[agent_name] = agent_info.agent_class
        
        logger.info(f"✅ Loaded {len(agents)} agents from RBA registry")
        return agents
        
    except Exception as e:
        logger.warning(f"⚠️ RBA registry failed: {e}, falling back to manual registry")
        # Fallback to manual registry
        return {
            'sandbagging_detection': SandbaggingRBAAgent,
            'pipeline_summary': PipelineSummaryRBAAgent,
        }

# Get dynamic agent registry - will be initialized at startup
AGENT_REGISTRY = {}

async def initialize_agent_registry():
    """Initialize the agent registry at startup"""
    global AGENT_REGISTRY
    try:
        AGENT_REGISTRY = await get_agent_registry()
        logger.info(f"✅ RBA Agent Registry initialized with {len(AGENT_REGISTRY)} agents")
    except Exception as e:
        logger.warning(f"⚠️ RBA registry failed: {e}, falling back to manual registry")
        AGENT_REGISTRY = get_fallback_registry()

# Configuration schemas for each agent
CONFIG_SCHEMAS = {
    'sandbagging_detection': {
        'agent_name': 'sandbagging_detection',
        'category': 'risk_analysis',
        'description': 'Identify sandbagging or inflated deals (low probability but high value)',
        'parameter_groups': {
            'Core Thresholds': [
                {
                    'name': 'high_value_threshold',
                    'type': 'currency',
                    'label': 'High Value Threshold',
                    'description': 'Minimum deal amount to be considered high-value',
                    'default': 250000,
                    'min': 10000,
                    'max': 50000000,
                    'ui_component': 'currency_input'
                },
                {
                    'name': 'sandbagging_threshold',
                    'type': 'percentage',
                    'label': 'Sandbagging Score Threshold',
                    'description': 'Minimum score to flag a deal as sandbagging',
                    'default': 65,
                    'min': 0,
                    'max': 100,
                    'ui_component': 'slider'
                },
                {
                    'name': 'low_probability_threshold',
                    'type': 'percentage',
                    'label': 'Low Probability Threshold',
                    'description': 'Maximum probability to be considered low',
                    'default': 35,
                    'min': 0,
                    'max': 100,
                    'ui_component': 'slider'
                }
            ],
            'Advanced Settings': [
                {
                    'name': 'advanced_stage_probability_threshold',
                    'type': 'percentage',
                    'label': 'Advanced Stage Min Probability',
                    'description': 'Minimum probability expected for advanced stages',
                    'default': 40,
                    'min': 0,
                    'max': 100,
                    'ui_component': 'slider'
                },
                {
                    'name': 'mega_deal_threshold',
                    'type': 'currency',
                    'label': 'Mega Deal Threshold',
                    'description': 'Amount threshold for mega deal classification',
                    'default': 1000000,
                    'min': 100000,
                    'max': 100000000,
                    'ui_component': 'currency_input'
                },
                {
                    'name': 'mega_deal_probability_threshold',
                    'type': 'percentage',
                    'label': 'Mega Deal Min Probability',
                    'description': 'Minimum probability required for mega deals',
                    'default': 50,
                    'min': 0,
                    'max': 100,
                    'ui_component': 'slider'
                },
                {
                    'name': 'confidence_threshold',
                    'type': 'percentage',
                    'label': 'Confidence Threshold',
                    'description': 'Minimum confidence level for flagging',
                    'default': 70,
                    'min': 50,
                    'max': 100,
                    'ui_component': 'slider'
                }
            ],
            'Industry Adjustments': [
                {
                    'name': 'enable_industry_adjustments',
                    'type': 'boolean',
                    'label': 'Enable Industry Adjustments',
                    'description': 'Apply industry-specific probability adjustments',
                    'default': True,
                    'ui_component': 'checkbox'
                }
            ]
        },
        'templates': [
            {
                'name': 'Conservative',
                'description': 'Higher thresholds, lower false positives',
                'config': {
                    'high_value_threshold': 500000,
                    'sandbagging_threshold': 80,
                    'low_probability_threshold': 25,
                    'confidence_threshold': 85
                }
            },
            {
                'name': 'Aggressive',
                'description': 'Lower thresholds, catch more potential issues',
                'config': {
                    'high_value_threshold': 100000,
                    'sandbagging_threshold': 50,
                    'low_probability_threshold': 45,
                    'confidence_threshold': 60
                }
            },
            {
                'name': 'Balanced',
                'description': 'Recommended settings for most use cases',
                'config': {
                    'high_value_threshold': 250000,
                    'sandbagging_threshold': 65,
                    'low_probability_threshold': 35,
                    'confidence_threshold': 70
                }
            }
        ],
        'result_schema': {
            'primary_visualizations': ['kpi_cards', 'risk_distribution_pie', 'top_deals_table'],
            'drill_down_fields': ['opportunity_id', 'account_id', 'owner_id'],
            'export_formats': ['csv', 'excel', 'pdf']
        }
    },
    
    'pipeline_summary': {
        'agent_name': 'pipeline_summary',
        'category': 'performance_analysis',
        'description': 'Generate comprehensive pipeline summary and overview metrics',
        'parameter_groups': {
            'Analysis Scope': [
                {
                    'name': 'include_closed_deals',
                    'type': 'boolean',
                    'label': 'Include Closed Deals',
                    'description': 'Include closed deals in analysis',
                    'default': True,
                    'ui_component': 'checkbox'
                },
                {
                    'name': 'include_stage_breakdown',
                    'type': 'boolean',
                    'label': 'Include Stage Breakdown',
                    'description': 'Show pipeline breakdown by stage',
                    'default': True,
                    'ui_component': 'checkbox'
                },
                {
                    'name': 'include_owner_breakdown',
                    'type': 'boolean',
                    'label': 'Include Owner Breakdown',
                    'description': 'Show pipeline breakdown by owner',
                    'default': True,
                    'ui_component': 'checkbox'
                }
            ],
            'Calculation Settings': [
                {
                    'name': 'probability_weighted',
                    'type': 'boolean',
                    'label': 'Probability Weighted',
                    'description': 'Use probability weighting in calculations',
                    'default': True,
                    'ui_component': 'checkbox'
                },
                {
                    'name': 'minimum_deal_amount',
                    'type': 'currency',
                    'label': 'Minimum Deal Amount',
                    'description': 'Minimum deal amount to include',
                    'default': 1000,
                    'min': 0,
                    'max': 1000000,
                    'ui_component': 'currency_input'
                }
            ]
        },
        'templates': [
            {
                'name': 'Executive Summary',
                'description': 'High-level overview for executives',
                'config': {
                    'include_closed_deals': False,
                    'probability_weighted': True,
                    'minimum_deal_amount': 50000
                }
            },
            {
                'name': 'Detailed Analysis',
                'description': 'Comprehensive analysis with all breakdowns',
                'config': {
                    'include_closed_deals': True,
                    'include_stage_breakdown': True,
                    'include_owner_breakdown': True,
                    'probability_weighted': True
                }
            }
        ],
        'result_schema': {
            'primary_visualizations': ['kpi_cards', 'pipeline_funnel', 'stage_breakdown_bar'],
            'drill_down_fields': ['opportunity_id', 'stage_name', 'owner_id'],
            'export_formats': ['csv', 'excel', 'pdf']
        }
    },
    
    # Additional agents with basic configurations
    'ownerless_deals': {
        'agent_name': 'ownerless_deals',
        'category': 'data_quality',
        'description': 'Detect ownerless or unassigned deals',
        'parameter_groups': {
            'Detection Criteria': [
                {
                    'name': 'check_inactive_owners',
                    'type': 'boolean',
                    'label': 'Check Inactive Owners',
                    'description': 'Flag deals assigned to inactive users',
                    'default': True,
                    'ui_component': 'toggle'
                },
                {
                    'name': 'minimum_deal_age_days',
                    'type': 'number',
                    'label': 'Minimum Deal Age (Days)',
                    'description': 'Only check deals older than this',
                    'default': 7,
                    'min': 1,
                    'max': 365,
                    'ui_component': 'slider'
                }
            ]
        },
        'templates': [
            {
                'name': 'Standard',
                'description': 'Default detection criteria',
                'config': {
                    'check_inactive_owners': True,
                    'minimum_deal_age_days': 7
                }
            },
            {
                'name': 'Strict',
                'description': 'More aggressive detection',
                'config': {
                    'check_inactive_owners': True,
                    'minimum_deal_age_days': 1
                }
            }
        ],
        'result_schema': {
            'primary_visualizations': ['kpi_cards', 'deals_table'],
            'drill_down_fields': ['opportunity_id', 'account_id'],
            'export_formats': ['csv', 'excel']
        }
    },
    
    'stage_velocity': {
        'agent_name': 'stage_velocity',
        'category': 'velocity_analysis',
        'description': 'Analyze stage velocity and deal progression',
        'parameter_groups': {
            'Velocity Thresholds': [
                {
                    'name': 'slow_velocity_days',
                    'type': 'number',
                    'label': 'Slow Velocity Threshold (Days)',
                    'description': 'Days in stage to be considered slow',
                    'default': 30,
                    'min': 7,
                    'max': 180,
                    'ui_component': 'slider'
                },
                {
                    'name': 'stuck_velocity_days',
                    'type': 'number',
                    'label': 'Stuck Velocity Threshold (Days)',
                    'description': 'Days in stage to be considered stuck',
                    'default': 60,
                    'min': 14,
                    'max': 365,
                    'ui_component': 'slider'
                }
            ]
        },
        'templates': [
            {
                'name': 'Conservative',
                'description': 'Longer time thresholds',
                'config': {
                    'slow_velocity_days': 45,
                    'stuck_velocity_days': 90
                }
            },
            {
                'name': 'Aggressive',
                'description': 'Shorter time thresholds',
                'config': {
                    'slow_velocity_days': 15,
                    'stuck_velocity_days': 30
                }
            }
        ],
        'result_schema': {
            'primary_visualizations': ['velocity_chart', 'stage_breakdown'],
            'drill_down_fields': ['opportunity_id', 'stage_name', 'days_in_stage'],
            'export_formats': ['csv', 'excel']
        }
    },
    
    'stale_deals': {
        'agent_name': 'stale_deals',
        'category': 'data_quality',
        'description': 'Identify stale deals stuck in pipeline stages',
        'parameter_groups': {
            'Stale Deal Criteria': [
                {
                    'name': 'stale_threshold_days',
                    'type': 'number',
                    'label': 'Stale Threshold (Days)',
                    'description': 'Days without activity to be considered stale',
                    'default': 60,
                    'min': 14,
                    'max': 180,
                    'ui_component': 'slider'
                },
                {
                    'name': 'minimum_deal_value',
                    'type': 'currency',
                    'label': 'Minimum Deal Value',
                    'description': 'Only check deals above this value',
                    'default': 50000,
                    'min': 1000,
                    'max': 10000000,
                    'ui_component': 'currency_input'
                }
            ]
        },
        'templates': [
            {
                'name': 'Standard',
                'description': '60-day stale threshold',
                'config': {
                    'stale_threshold_days': 60,
                    'minimum_deal_value': 50000
                }
            },
            {
                'name': 'Aggressive',
                'description': '30-day stale threshold',
                'config': {
                    'stale_threshold_days': 30,
                    'minimum_deal_value': 25000
                }
            }
        ],
        'result_schema': {
            'primary_visualizations': ['kpi_cards', 'stale_deals_table'],
            'drill_down_fields': ['opportunity_id', 'days_stale', 'last_activity'],
            'export_formats': ['csv', 'excel']
        }
    },
    
    'duplicate_detection': {
        'agent_name': 'duplicate_detection',
        'category': 'data_quality',
        'description': 'Detect duplicate deals across the pipeline',
        'parameter_groups': {
            'Matching Criteria': [
                {
                    'name': 'match_threshold',
                    'type': 'percentage',
                    'label': 'Match Threshold (%)',
                    'description': 'Similarity percentage to consider duplicates',
                    'default': 85,
                    'min': 50,
                    'max': 100,
                    'ui_component': 'percentage_input'
                },
                {
                    'name': 'check_account_name',
                    'type': 'boolean',
                    'label': 'Check Account Name',
                    'description': 'Include account name in duplicate matching',
                    'default': True,
                    'ui_component': 'toggle'
                }
            ]
        },
        'templates': [
            {
                'name': 'Strict',
                'description': 'High threshold for exact matches',
                'config': {
                    'match_threshold': 95,
                    'check_account_name': True
                }
            },
            {
                'name': 'Fuzzy',
                'description': 'Lower threshold for similar deals',
                'config': {
                    'match_threshold': 70,
                    'check_account_name': False
                }
            }
        ],
        'result_schema': {
            'primary_visualizations': ['duplicate_pairs_table', 'match_confidence_chart'],
            'drill_down_fields': ['opportunity_id', 'duplicate_id', 'match_score'],
            'export_formats': ['csv', 'excel']
        }
    },
    
    'missing_fields': {
        'agent_name': 'missing_fields',
        'category': 'data_quality',
        'description': 'Audit deals for missing required fields',
        'parameter_groups': {
            'Field Requirements': [
                {
                    'name': 'require_close_date',
                    'type': 'boolean',
                    'label': 'Require Close Date',
                    'description': 'Flag deals missing close dates',
                    'default': True,
                    'ui_component': 'toggle'
                },
                {
                    'name': 'require_amount',
                    'type': 'boolean',
                    'label': 'Require Deal Amount',
                    'description': 'Flag deals missing amounts',
                    'default': True,
                    'ui_component': 'toggle'
                }
            ]
        },
        'templates': [
            {
                'name': 'Basic',
                'description': 'Check core required fields only',
                'config': {
                    'require_close_date': True,
                    'require_amount': True
                }
            },
            {
                'name': 'Complete',
                'description': 'Check all recommended fields',
                'config': {
                    'require_close_date': True,
                    'require_amount': True
                }
            }
        ],
        'result_schema': {
            'primary_visualizations': ['missing_fields_chart', 'incomplete_deals_table'],
            'drill_down_fields': ['opportunity_id', 'missing_fields', 'completion_score'],
            'export_formats': ['csv', 'excel']
        }
    }
}

@router.get("/agents")
async def get_available_agents():
    """Get list of available RBA agents"""
    agents = []
    for agent_name, schema in CONFIG_SCHEMAS.items():
        agents.append({
            'name': agent_name,
            'category': schema['category'],
            'description': schema['description']
        })
    
    return {
        'success': True,
        'agents': agents,
        'total_count': len(agents)
    }

@router.get("/agents/{agent_name}/schema")
async def get_agent_config_schema(agent_name: str):
    """Get configuration schema for a specific agent"""
    if agent_name not in CONFIG_SCHEMAS:
        raise HTTPException(status_code=404, detail=f"Agent '{agent_name}' not found")
    
    schema = CONFIG_SCHEMAS[agent_name]
    
    # Add parameter count for UI
    param_count = sum(len(group) for group in schema['parameter_groups'].values())
    schema['parameter_count'] = param_count
    
    return {
        'success': True,
        'schema': schema
    }

@router.post("/agents/{agent_name}/validate")
async def validate_agent_config(agent_name: str, config: Dict[str, Any] = Body(...)):
    """Validate configuration for a specific agent"""
    if agent_name not in CONFIG_SCHEMAS:
        raise HTTPException(status_code=404, detail=f"Agent '{agent_name}' not found")
    
    schema = CONFIG_SCHEMAS[agent_name]
    errors = []
    validated_config = {}
    
    # Get all parameters from schema
    all_params = {}
    for group in schema['parameter_groups'].values():
        for param in group:
            all_params[param['name']] = param
    
    # Validate each parameter
    for param_name, param_def in all_params.items():
        value = config.get(param_name, param_def['default'])
        
        try:
            # Type validation and conversion
            if param_def['type'] == 'currency':
                validated_value = float(value)
                if 'min' in param_def and validated_value < param_def['min']:
                    errors.append(f"{param_name}: Value {validated_value} below minimum {param_def['min']}")
                if 'max' in param_def and validated_value > param_def['max']:
                    errors.append(f"{param_name}: Value {validated_value} above maximum {param_def['max']}")
            elif param_def['type'] == 'percentage':
                validated_value = float(value)
                if validated_value < 0 or validated_value > 100:
                    errors.append(f"{param_name}: Percentage must be between 0 and 100")
            elif param_def['type'] == 'boolean':
                validated_value = bool(value)
            else:
                validated_value = value
            
            validated_config[param_name] = validated_value
            
        except (ValueError, TypeError) as e:
            errors.append(f"{param_name}: Invalid value '{value}' - {str(e)}")
            validated_config[param_name] = param_def['default']
    
    return {
        'success': len(errors) == 0,
        'validated_config': validated_config,
        'errors': errors
    }

@router.post("/agents/{agent_name}/preview")
async def preview_config_impact(agent_name: str, config: Dict[str, Any] = Body(...)):
    """Preview the impact of configuration changes"""
    if agent_name not in CONFIG_SCHEMAS:
        raise HTTPException(status_code=404, detail=f"Agent '{agent_name}' not found")
    
    # Get default config
    schema = CONFIG_SCHEMAS[agent_name]
    default_config = {}
    for group in schema['parameter_groups'].values():
        for param in group:
            default_config[param['name']] = param['default']
    
    # Analyze changes
    changes = []
    for param_name, new_value in config.items():
        if param_name in default_config:
            old_value = default_config[param_name]
            if new_value != old_value:
                if isinstance(old_value, (int, float)) and isinstance(new_value, (int, float)):
                    change_pct = ((new_value - old_value) / old_value * 100) if old_value != 0 else 0
                    changes.append({
                        'parameter': param_name,
                        'old_value': old_value,
                        'new_value': new_value,
                        'change_percentage': round(change_pct, 1),
                        'impact': 'higher sensitivity' if change_pct < 0 else 'lower sensitivity'
                    })
    
    # Generate preview
    impact_level = 'low'
    if len(changes) > 3:
        impact_level = 'high'
    elif len(changes) > 1:
        impact_level = 'medium'
    
    return {
        'success': True,
        'impact_level': impact_level,
        'changes': changes,
        'recommendations': [
            "Test with a small dataset first" if impact_level == 'high' else "Changes look reasonable",
            f"Expect {'more' if impact_level == 'high' else 'similar'} results with current settings"
        ]
    }

@router.get("/agents/{agent_name}/templates")
async def get_agent_templates(agent_name: str):
    """Get configuration templates for an agent"""
    if agent_name not in CONFIG_SCHEMAS:
        raise HTTPException(status_code=404, detail=f"Agent '{agent_name}' not found")
    
    templates = CONFIG_SCHEMAS[agent_name].get('templates', [])
    
    return {
        'success': True,
        'templates': templates
    }

@router.post("/agents/{agent_name}/apply-template")
async def apply_config_template(agent_name: str, template_name: str = Body(...)):
    """Apply a configuration template"""
    if agent_name not in CONFIG_SCHEMAS:
        raise HTTPException(status_code=404, detail=f"Agent '{agent_name}' not found")
    
    templates = CONFIG_SCHEMAS[agent_name].get('templates', [])
    template = next((t for t in templates if t['name'] == template_name), None)
    
    if not template:
        raise HTTPException(status_code=404, detail=f"Template '{template_name}' not found")
    
    return {
        'success': True,
        'config': template['config'],
        'description': template['description']
    }

@router.get("/categories")
async def get_agent_categories():
    """Get all agent categories with counts"""
    categories = {}
    
    for agent_name, schema in CONFIG_SCHEMAS.items():
        category = schema['category']
        if category not in categories:
            categories[category] = {
                'name': category,
                'display_name': category.replace('_', ' ').title(),
                'agents': []
            }
        categories[category]['agents'].append({
            'name': agent_name,
            'description': schema['description']
        })
    
    return {
        'success': True,
        'categories': list(categories.values())
    }

@router.get("/health")
async def config_api_health():
    """Health check for configuration API"""
    return {
        'success': True,
        'status': 'healthy',
        'available_agents': len(CONFIG_SCHEMAS),
        'timestamp': datetime.now().isoformat()
    }
