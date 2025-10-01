"""
Dynamic Capability Registry Module
=================================

This module provides dynamic, adaptive SaaS automation capabilities including:
- Pattern discovery from business data
- Dynamic template generation
- Intelligent recommendations
- Continuous learning and adaptation
"""

# Core dynamic capability components
from .dynamic_saas_engine import DynamicSaaSCapabilityEngine, SaaSDataPattern, SaaSBusinessModel, SaaSMetricCategory
from .saas_template_generators import SaaSRBATemplateGenerator, SaaSRBIATemplateGenerator, SaaSAALATemplateGenerator
from .dynamic_capability_orchestrator import DynamicCapabilityOrchestrator

# Note: schema_api is not imported here to avoid dependency issues
# It will be imported separately when needed

__all__ = [
    'DynamicSaaSCapabilityEngine',
    'SaaSDataPattern', 
    'SaaSBusinessModel',
    'SaaSMetricCategory',
    'SaaSRBATemplateGenerator',
    'SaaSRBIATemplateGenerator', 
    'SaaSAALATemplateGenerator',
    'DynamicCapabilityOrchestrator'
]