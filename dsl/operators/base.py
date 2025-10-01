"""
Base Operator Class - Foundation for all DSL operators
"""

import asyncio
import time
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional, Union
from datetime import datetime
import logging

logger = logging.getLogger(__name__)

@dataclass
class OperatorContext:
    """Context passed to operators during execution"""
    # User context - required fields first
    user_id: int
    tenant_id: str
    workflow_id: str
    step_id: str
    execution_id: str
    
    # Optional fields with defaults
    session_id: Optional[str] = None
    input_data: Dict[str, Any] = field(default_factory=dict)
    previous_outputs: Dict[str, Any] = field(default_factory=dict)
    policy_pack_id: Optional[str] = None
    trust_threshold: float = 0.70
    evidence_required: bool = True
    pool_manager: Any = None
    
    def get_tenant_setting(self, key: str, default: Any = None) -> Any:
        """Get tenant-specific setting with fallback to default"""
        # This will be implemented to fetch from tenant configuration
        return default

@dataclass
class OperatorResult:
    """Result returned by operator execution"""
    # Execution results
    success: bool
    output_data: Dict[str, Any] = field(default_factory=dict)
    error_message: Optional[str] = None
    
    # Performance metrics
    execution_time_ms: int = 0
    confidence_score: float = 1.0
    
    # Governance
    evidence_data: Dict[str, Any] = field(default_factory=dict)
    policy_violations: List[str] = field(default_factory=list)
    
    # Next step routing
    next_step_id: Optional[str] = None
    
    def add_evidence(self, key: str, value: Any):
        """Add evidence data for governance"""
        self.evidence_data[key] = value
    
    def add_violation(self, violation: str):
        """Add policy violation"""
        self.policy_violations.append(violation)

class BaseOperator(ABC):
    """
    Abstract base class for all DSL operators
    
    All operators must implement:
    - validate_config: Validate operator configuration
    - execute_async: Perform the actual operation
    """
    
    def __init__(self, operator_id: str):
        self.operator_id = operator_id
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
    
    @abstractmethod
    async def validate_config(self, config: Dict[str, Any]) -> List[str]:
        """
        Validate operator configuration
        
        Args:
            config: Operator configuration from DSL
            
        Returns:
            List of validation errors (empty if valid)
        """
        pass
    
    @abstractmethod
    async def execute_async(self, context: OperatorContext, config: Dict[str, Any]) -> OperatorResult:
        """
        Execute the operator asynchronously
        
        Args:
            context: Execution context
            config: Operator configuration
            
        Returns:
            OperatorResult with execution results
        """
        pass
    
    async def execute(self, context: OperatorContext, config: Dict[str, Any]) -> OperatorResult:
        """
        Main execution wrapper with governance and performance tracking
        """
        start_time = time.time()
        execution_id = str(uuid.uuid4())
        
        try:
            # Pre-execution validation
            validation_errors = await self.validate_config(config)
            if validation_errors:
                return OperatorResult(
                    success=False,
                    error_message=f"Configuration validation failed: {', '.join(validation_errors)}",
                    execution_time_ms=int((time.time() - start_time) * 1000)
                )
            
            # Governance checks
            governance_result = await self._check_governance(context, config)
            if not governance_result.success:
                return governance_result
            
            # Execute the actual operation
            self.logger.info(f"Executing {self.__class__.__name__} for step {context.step_id}")
            result = await self.execute_async(context, config)
            
            # Post-execution processing
            execution_time = int((time.time() - start_time) * 1000)
            result.execution_time_ms = execution_time
            
            # Add default evidence
            result.add_evidence('operator_type', self.__class__.__name__)
            result.add_evidence('execution_id', execution_id)
            result.add_evidence('execution_time_ms', execution_time)
            result.add_evidence('tenant_id', context.tenant_id)
            
            # Governance evidence
            if context.evidence_required:
                await self._generate_evidence(context, config, result)
            
            self.logger.info(f"Completed {self.__class__.__name__} in {execution_time}ms")
            return result
            
        except Exception as e:
            execution_time = int((time.time() - start_time) * 1000)
            self.logger.error(f"Error in {self.__class__.__name__}: {e}")
            
            return OperatorResult(
                success=False,
                error_message=str(e),
                execution_time_ms=execution_time,
                evidence_data={
                    'operator_type': self.__class__.__name__,
                    'execution_id': execution_id,
                    'error': str(e),
                    'tenant_id': context.tenant_id
                }
            )
    
    async def _check_governance(self, context: OperatorContext, config: Dict[str, Any]) -> OperatorResult:
        """Check governance policies before execution"""
        try:
            # Policy validation will be implemented here
            # For now, basic checks
            
            # Check tenant isolation
            if not context.tenant_id:
                return OperatorResult(
                    success=False,
                    error_message="Tenant ID required for governance"
                )
            
            # Check trust threshold for ML/Agent operations
            if hasattr(self, 'requires_trust_check') and self.requires_trust_check:
                if context.trust_threshold < 0.5:
                    return OperatorResult(
                        success=False,
                        error_message="Trust threshold too low for this operation"
                    )
            
            return OperatorResult(success=True)
            
        except Exception as e:
            return OperatorResult(
                success=False,
                error_message=f"Governance check failed: {e}"
            )
    
    async def _generate_evidence(self, context: OperatorContext, config: Dict[str, Any], result: OperatorResult):
        """Generate evidence data for audit trails"""
        try:
            # Add governance evidence
            result.add_evidence('governance_check_passed', True)
            result.add_evidence('policy_pack_id', context.policy_pack_id)
            result.add_evidence('user_id', context.user_id)
            result.add_evidence('timestamp', datetime.utcnow().isoformat())
            
            # Add configuration evidence (sanitized)
            sanitized_config = self._sanitize_config_for_evidence(config)
            result.add_evidence('operator_config', sanitized_config)
            
        except Exception as e:
            self.logger.warning(f"Failed to generate evidence: {e}")
    
    def _sanitize_config_for_evidence(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Remove sensitive data from config for evidence"""
        # Remove common sensitive fields
        sensitive_fields = ['password', 'token', 'secret', 'key']
        sanitized = {}
        
        for key, value in config.items():
            if any(sensitive in key.lower() for sensitive in sensitive_fields):
                sanitized[key] = "[REDACTED]"
            else:
                sanitized[key] = value
                
        return sanitized
