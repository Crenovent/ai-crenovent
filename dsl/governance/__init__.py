"""
DSL Governance Module
====================
Task 7.1.3: Configure mandatory governance metadata fields

This module provides governance-by-design validation for DSL workflows.
All validation rules are configuration-driven and support multiple enforcement modes.

Key Features:
- Configuration-driven validation (no hardcoding)
- Multi-tenant isolation enforcement
- Industry overlay support
- Compliance framework integration
- Dynamic validation rules engine
- Modular enforcement modes
"""

from .governance_metadata_validator import (
    GovernanceMetadataValidator,
    get_governance_validator,
    ValidationError,
    ValidationResult,
    ValidationContext,
    EnforcementMode
)

from .multi_tenant_taxonomy import (
    MultiTenantEnforcementEngine
)

from .policy_engine import (
    PolicyEngine,
    PolicyPack,
    PolicyViolation,
    OverrideRequest,
    PolicyPackType,
    EnforcementLevel,
    OverrideType
)

__all__ = [
    'GovernanceMetadataValidator',
    'get_governance_validator',
    'ValidationError',
    'ValidationResult', 
    'ValidationContext',
    'EnforcementMode',
    'MultiTenantEnforcementEngine',
    'PolicyEngine',
    'PolicyPack',
    'PolicyViolation',
    'OverrideRequest',
    'PolicyPackType',
    'EnforcementLevel',
    'OverrideType'
]