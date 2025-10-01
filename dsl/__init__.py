"""
DSL (Domain Specific Language) Foundation for RevAI Pro
Enterprise-grade automation backbone with governance-first design
"""

__version__ = "1.0.0"
__author__ = "Crenovent AI Team"

from .parser import DSLParser
from .orchestrator import DSLOrchestrator
from .operators import *
from .governance import MultiTenantEnforcementEngine
from .storage import WorkflowStorage

__all__ = [
    'DSLParser',
    'DSLOrchestrator', 
    'MultiTenantEnforcementEngine',
    'WorkflowStorage'
]
