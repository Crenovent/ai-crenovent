"""
Task 8.1-T23: Add residency overlay enforcement (EU data cannot run in US infra)
Geographic data residency enforcement for compliance
"""

import asyncio
import json
import logging
import uuid
from datetime import datetime, timezone
from typing import Dict, List, Any, Optional, Set
from dataclasses import dataclass, field
from enum import Enum

import asyncpg

logger = logging.getLogger(__name__)


class DataResidencyRegion(Enum):
    """Supported data residency regions"""
    EU = "eu"
    US = "us"
    INDIA = "in"
    CANADA = "ca"
    AUSTRALIA = "au"
    SINGAPORE = "sg"
    JAPAN = "jp"
    UK = "uk"


class ResidencyViolationType(Enum):
    """Types of residency violations"""
    CROSS_BORDER_TRANSFER = "cross_border_transfer"
    UNAUTHORIZED_REGION = "unauthorized_region"
    MISSING_RESIDENCY_TAG = "missing_residency_tag"
    INVALID_INFRASTRUCTURE = "invalid_infrastructure"


@dataclass
class ResidencyPolicy:
    """Data residency policy definition"""
    policy_id: str
    policy_name: str
    
    # Residency rules
    allowed_regions: List[DataResidencyRegion] = field(default_factory=list)
    prohibited_regions: List[DataResidencyRegion] = field(default_factory=list)
    
    # Data classification
    data_types: List[str] = field(default_factory=list)  # PII, PHI, financial, etc.
    sensitivity_levels: List[str] = field(default_factory=list)  # public, internal, confidential, restricted
    
    # Compliance requirements
    compliance_frameworks: List[str] = field(default_factory=list)  # GDPR, HIPAA, etc.
    industry_overlay: Optional[str] = None
    
    # Enforcement settings
    strict_enforcement: bool = True
    allow_temporary_transfer: bool = False
    max_transfer_duration_hours: int = 0
    
    # Metadata
    description: str = ""
    is_active: bool = True
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "policy_id": self.policy_id,
            "policy_name": self.policy_name,
            "allowed_regions": [r.value for r in self.allowed_regions],
            "prohibited_regions": [r.value for r in self.prohibited_regions],
            "data_types": self.data_types,
            "sensitivity_levels": self.sensitivity_levels,
            "compliance_frameworks": self.compliance_frameworks,
            "industry_overlay": self.industry_overlay,
            "strict_enforcement": self.strict_enforcement,
            "allow_temporary_transfer": self.allow_temporary_transfer,
            "max_transfer_duration_hours": self.max_transfer_duration_hours,
            "description": self.description,
            "is_active": self.is_active,
            "created_at": self.created_at.isoformat()
        }


@dataclass
class ResidencyViolation:
    """Data residency violation"""
    violation_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    policy_id: str = ""
    
    # Violation details
    violation_type: ResidencyViolationType = ResidencyViolationType.CROSS_BORDER_TRANSFER
    severity: str = "high"
    message: str = ""
    
    # Context
    workflow_id: str = ""
    execution_id: str = ""
    data_type: str = ""
    
    # Geographic details
    source_region: Optional[DataResidencyRegion] = None
    target_region: Optional[DataResidencyRegion] = None
    current_infrastructure_region: Optional[DataResidencyRegion] = None
    
    # Timestamps
    detected_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "violation_id": self.violation_id,
            "policy_id": self.policy_id,
            "violation_type": self.violation_type.value,
            "severity": self.severity,
            "message": self.message,
            "workflow_id": self.workflow_id,
            "execution_id": self.execution_id,
            "data_type": self.data_type,
            "source_region": self.source_region.value if self.source_region else None,
            "target_region": self.target_region.value if self.target_region else None,
            "current_infrastructure_region": self.current_infrastructure_region.value if self.current_infrastructure_region else None,
            "detected_at": self.detected_at.isoformat()
        }


@dataclass
class ResidencyEnforcementResult:
    """Result of residency enforcement check"""
    execution_id: str
    is_compliant: bool = True
    violations: List[ResidencyViolation] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    
    # Enforcement actions
    blocked_operations: List[str] = field(default_factory=list)
    allowed_operations: List[str] = field(default_factory=list)
    
    # Performance
    enforcement_duration_ms: float = 0.0
    
    # Timestamps
    enforced_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "execution_id": self.execution_id,
            "is_compliant": self.is_compliant,
            "violations": [v.to_dict() for v in self.violations],
            "warnings": self.warnings,
            "blocked_operations": self.blocked_operations,
            "allowed_operations": self.allowed_operations,
            "enforcement_duration_ms": self.enforcement_duration_ms,
            "enforced_at": self.enforced_at.isoformat()
        }


class ResidencyPolicyManager:
    """
    Residency Policy Manager
    Manages data residency policies and rules
    """
    
    def __init__(self):
        self.policies: Dict[str, ResidencyPolicy] = {}
        self._initialize_default_policies()
    
    def _initialize_default_policies(self):
        """Initialize default residency policies"""
        
        # GDPR EU data residency policy
        gdpr_policy = ResidencyPolicy(
            policy_id="gdpr_eu_residency",
            policy_name="GDPR EU Data Residency",
            allowed_regions=[DataResidencyRegion.EU, DataResidencyRegion.UK],
            prohibited_regions=[DataResidencyRegion.US, DataResidencyRegion.INDIA, DataResidencyRegion.SINGAPORE],
            data_types=["personal_data", "pii", "customer_data"],
            sensitivity_levels=["confidential", "restricted"],
            compliance_frameworks=["GDPR"],
            industry_overlay="SaaS",
            strict_enforcement=True,
            description="GDPR requires EU personal data to remain within EU/EEA"
        )
        
        # HIPAA US healthcare data policy
        hipaa_policy = ResidencyPolicy(
            policy_id="hipaa_us_residency",
            policy_name="HIPAA US Healthcare Data Residency",
            allowed_regions=[DataResidencyRegion.US, DataResidencyRegion.CANADA],
            prohibited_regions=[DataResidencyRegion.EU, DataResidencyRegion.INDIA, DataResidencyRegion.SINGAPORE],
            data_types=["phi", "medical_data", "patient_data"],
            sensitivity_levels=["restricted"],
            compliance_frameworks=["HIPAA"],
            industry_overlay="Healthcare",
            strict_enforcement=True,
            description="HIPAA requires US healthcare data to remain in approved regions"
        )
        
        # RBI India banking data policy
        rbi_policy = ResidencyPolicy(
            policy_id="rbi_india_residency",
            policy_name="RBI India Banking Data Residency",
            allowed_regions=[DataResidencyRegion.INDIA],
            prohibited_regions=[DataResidencyRegion.US, DataResidencyRegion.EU, DataResidencyRegion.SINGAPORE],
            data_types=["financial_data", "banking_data", "payment_data"],
            sensitivity_levels=["confidential", "restricted"],
            compliance_frameworks=["RBI"],
            industry_overlay="Banking",
            strict_enforcement=True,
            description="RBI requires Indian banking data to remain within India"
        )
        
        # SaaS general data policy
        saas_policy = ResidencyPolicy(
            policy_id="saas_flexible_residency",
            policy_name="SaaS Flexible Data Residency",
            allowed_regions=[DataResidencyRegion.US, DataResidencyRegion.EU, DataResidencyRegion.CANADA],
            prohibited_regions=[],
            data_types=["business_data", "analytics_data"],
            sensitivity_levels=["public", "internal"],
            compliance_frameworks=["SOX"],
            industry_overlay="SaaS",
            strict_enforcement=False,
            allow_temporary_transfer=True,
            max_transfer_duration_hours=24,
            description="Flexible residency for non-sensitive SaaS data"
        )
        
        self.policies = {
            policy.policy_id: policy for policy in [
                gdpr_policy, hipaa_policy, rbi_policy, saas_policy
            ]
        }
    
    def get_applicable_policies(
        self,
        data_type: str,
        sensitivity_level: str,
        compliance_frameworks: List[str],
        industry_overlay: str
    ) -> List[ResidencyPolicy]:
        """Get applicable residency policies"""
        
        applicable = []
        
        for policy in self.policies.values():
            if not policy.is_active:
                continue
            
            # Check data type
            if policy.data_types and data_type not in policy.data_types:
                continue
            
            # Check sensitivity level
            if policy.sensitivity_levels and sensitivity_level not in policy.sensitivity_levels:
                continue
            
            # Check compliance frameworks
            if policy.compliance_frameworks:
                if not any(cf in compliance_frameworks for cf in policy.compliance_frameworks):
                    continue
            
            # Check industry overlay
            if policy.industry_overlay and policy.industry_overlay != industry_overlay:
                continue
            
            applicable.append(policy)
        
        return applicable


class ResidencyEnforcementEngine:
    """
    Residency Enforcement Engine - Task 8.1-T23
    
    Enforces data residency policies to prevent cross-border data transfers
    Ensures compliance with GDPR, HIPAA, RBI and other regional regulations
    """
    
    def __init__(self, db_pool: Optional[asyncpg.Pool] = None):
        self.db_pool = db_pool
        self.logger = logging.getLogger(__name__)
        
        # Configuration
        self.config = {
            'strict_enforcement': True,
            'fail_closed': True,
            'enable_audit_logging': True,
            'block_violations': True,
            'alert_on_violations': True
        }
        
        # Policy manager
        self.policy_manager = ResidencyPolicyManager()
        
        # Infrastructure region mapping
        self.infrastructure_regions = {
            'us-east-1': DataResidencyRegion.US,
            'us-west-2': DataResidencyRegion.US,
            'eu-west-1': DataResidencyRegion.EU,
            'eu-central-1': DataResidencyRegion.EU,
            'ap-south-1': DataResidencyRegion.INDIA,
            'ca-central-1': DataResidencyRegion.CANADA,
            'ap-southeast-1': DataResidencyRegion.SINGAPORE,
            'ap-northeast-1': DataResidencyRegion.JAPAN,
            'eu-west-2': DataResidencyRegion.UK,
            'ap-southeast-2': DataResidencyRegion.AUSTRALIA
        }
        
        # Statistics
        self.enforcement_stats = {
            'total_checks': 0,
            'compliant_operations': 0,
            'violations_detected': 0,
            'operations_blocked': 0,
            'cross_border_transfers_prevented': 0,
            'average_enforcement_time_ms': 0.0
        }
    
    async def initialize(self) -> bool:
        """Initialize residency enforcement engine"""
        try:
            await self._create_residency_tables()
            self.logger.info("✅ Residency enforcement engine initialized")
            return True
        except Exception as e:
            self.logger.error(f"❌ Failed to initialize residency enforcement engine: {e}")
            return False
    
    async def enforce_data_residency(
        self,
        execution_context: Dict[str, Any],
        tenant_id: int,
        current_infrastructure_region: str = "us-east-1"
    ) -> ResidencyEnforcementResult:
        """
        Enforce data residency policies for workflow execution
        
        This is the main entry point called by the runtime
        """
        
        start_time = datetime.now(timezone.utc)
        
        execution_id = execution_context.get('execution_id', 'unknown')
        workflow_id = execution_context.get('workflow_id', 'unknown')
        
        # Initialize result
        result = ResidencyEnforcementResult(execution_id=execution_id)
        
        try:
            # Get current infrastructure region
            current_region = self.infrastructure_regions.get(
                current_infrastructure_region, DataResidencyRegion.US
            )
            
            # Extract data context
            data_operations = self._extract_data_operations(execution_context)
            
            # Check each data operation
            for operation in data_operations:
                violations = await self._check_data_operation_residency(
                    operation, current_region, workflow_id, execution_id, tenant_id
                )
                result.violations.extend(violations)
                
                if violations:
                    # Block operation if violations found
                    if self.config['block_violations']:
                        result.blocked_operations.append(operation['operation_id'])
                        self.logger.warning(f"🚫 Blocked operation {operation['operation_id']} due to residency violation")
                    else:
                        result.warnings.append(f"Residency violation in operation {operation['operation_id']}")
                else:
                    result.allowed_operations.append(operation['operation_id'])
            
            # Determine overall compliance
            if result.violations and self.config['strict_enforcement']:
                result.is_compliant = False
            
            # Calculate enforcement duration
            end_time = datetime.now(timezone.utc)
            result.enforcement_duration_ms = (end_time - start_time).total_seconds() * 1000
            
            # Update statistics
            self._update_enforcement_stats(result)
            
            # Store enforcement result
            if self.db_pool and self.config['enable_audit_logging']:
                await self._store_enforcement_result(result, tenant_id)
            
            if result.is_compliant:
                self.logger.info(f"✅ Data residency compliance verified for execution {execution_id}")
            else:
                self.logger.error(f"❌ Data residency violations detected for execution {execution_id}")
                for violation in result.violations:
                    self.logger.error(f"🚫 {violation.violation_type.value}: {violation.message}")
            
            return result
            
        except Exception as e:
            self.logger.error(f"❌ Residency enforcement error: {e}")
            
            # Fail closed on enforcement errors
            if self.config['fail_closed']:
                result.is_compliant = False
                result.violations.append(ResidencyViolation(
                    policy_id="system_error",
                    violation_type=ResidencyViolationType.UNAUTHORIZED_REGION,
                    severity="critical",
                    message=f"Residency enforcement system error: {str(e)}",
                    workflow_id=workflow_id,
                    execution_id=execution_id
                ))
            
            result.enforcement_duration_ms = (datetime.now(timezone.utc) - start_time).total_seconds() * 1000
            return result
    
    def _extract_data_operations(self, execution_context: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Extract data operations from execution context"""
        
        operations = []
        
        # Extract from workflow steps
        steps = execution_context.get('workflow_steps', [])
        for i, step in enumerate(steps):
            if step.get('type') in ['query', 'data_processing', 'ml_decision']:
                operation = {
                    'operation_id': f"step_{i}_{step.get('id', 'unknown')}",
                    'operation_type': step.get('type'),
                    'data_sources': step.get('data_sources', []),
                    'data_types': step.get('data_types', ['business_data']),
                    'sensitivity_level': step.get('sensitivity_level', 'internal'),
                    'target_regions': step.get('target_regions', []),
                    'compliance_frameworks': step.get('compliance_frameworks', [])
                }
                operations.append(operation)
        
        # Extract from data context
        data_context = execution_context.get('data_context', {})
        if data_context:
            operation = {
                'operation_id': 'data_context',
                'operation_type': 'data_access',
                'data_sources': data_context.get('sources', []),
                'data_types': data_context.get('types', ['business_data']),
                'sensitivity_level': data_context.get('sensitivity', 'internal'),
                'target_regions': data_context.get('regions', []),
                'compliance_frameworks': data_context.get('compliance', [])
            }
            operations.append(operation)
        
        return operations
    
    async def _check_data_operation_residency(
        self,
        operation: Dict[str, Any],
        current_region: DataResidencyRegion,
        workflow_id: str,
        execution_id: str,
        tenant_id: int
    ) -> List[ResidencyViolation]:
        """Check data operation against residency policies"""
        
        violations = []
        
        # Get applicable policies
        applicable_policies = self.policy_manager.get_applicable_policies(
            data_type=operation.get('data_types', ['business_data'])[0],
            sensitivity_level=operation.get('sensitivity_level', 'internal'),
            compliance_frameworks=operation.get('compliance_frameworks', []),
            industry_overlay=operation.get('industry_overlay', 'SaaS')
        )
        
        for policy in applicable_policies:
            violation = await self._validate_against_policy(
                operation, policy, current_region, workflow_id, execution_id
            )
            if violation:
                violations.append(violation)
        
        return violations
    
    async def _validate_against_policy(
        self,
        operation: Dict[str, Any],
        policy: ResidencyPolicy,
        current_region: DataResidencyRegion,
        workflow_id: str,
        execution_id: str
    ) -> Optional[ResidencyViolation]:
        """Validate operation against specific residency policy"""
        
        # Check if current region is allowed
        if policy.allowed_regions and current_region not in policy.allowed_regions:
            return ResidencyViolation(
                policy_id=policy.policy_id,
                violation_type=ResidencyViolationType.UNAUTHORIZED_REGION,
                severity="critical" if policy.strict_enforcement else "high",
                message=f"Data operation not allowed in region {current_region.value}. Policy allows: {[r.value for r in policy.allowed_regions]}",
                workflow_id=workflow_id,
                execution_id=execution_id,
                data_type=operation.get('data_types', ['unknown'])[0],
                current_infrastructure_region=current_region
            )
        
        # Check if current region is prohibited
        if policy.prohibited_regions and current_region in policy.prohibited_regions:
            return ResidencyViolation(
                policy_id=policy.policy_id,
                violation_type=ResidencyViolationType.UNAUTHORIZED_REGION,
                severity="critical",
                message=f"Data operation prohibited in region {current_region.value}. Policy prohibits: {[r.value for r in policy.prohibited_regions]}",
                workflow_id=workflow_id,
                execution_id=execution_id,
                data_type=operation.get('data_types', ['unknown'])[0],
                current_infrastructure_region=current_region
            )
        
        # Check for cross-border transfers
        target_regions = operation.get('target_regions', [])
        for target_region_str in target_regions:
            try:
                target_region = DataResidencyRegion(target_region_str)
                
                if policy.allowed_regions and target_region not in policy.allowed_regions:
                    return ResidencyViolation(
                        policy_id=policy.policy_id,
                        violation_type=ResidencyViolationType.CROSS_BORDER_TRANSFER,
                        severity="critical",
                        message=f"Cross-border transfer to {target_region.value} not allowed by policy {policy.policy_name}",
                        workflow_id=workflow_id,
                        execution_id=execution_id,
                        data_type=operation.get('data_types', ['unknown'])[0],
                        source_region=current_region,
                        target_region=target_region
                    )
                
                if policy.prohibited_regions and target_region in policy.prohibited_regions:
                    return ResidencyViolation(
                        policy_id=policy.policy_id,
                        violation_type=ResidencyViolationType.CROSS_BORDER_TRANSFER,
                        severity="critical",
                        message=f"Cross-border transfer to prohibited region {target_region.value}",
                        workflow_id=workflow_id,
                        execution_id=execution_id,
                        data_type=operation.get('data_types', ['unknown'])[0],
                        source_region=current_region,
                        target_region=target_region
                    )
            except ValueError:
                # Invalid region string
                return ResidencyViolation(
                    policy_id=policy.policy_id,
                    violation_type=ResidencyViolationType.INVALID_INFRASTRUCTURE,
                    severity="medium",
                    message=f"Invalid target region specified: {target_region_str}",
                    workflow_id=workflow_id,
                    execution_id=execution_id,
                    data_type=operation.get('data_types', ['unknown'])[0]
                )
        
        return None
    
    def _update_enforcement_stats(self, result: ResidencyEnforcementResult):
        """Update enforcement statistics"""
        
        self.enforcement_stats['total_checks'] += 1
        
        if result.is_compliant:
            self.enforcement_stats['compliant_operations'] += 1
        else:
            self.enforcement_stats['violations_detected'] += len(result.violations)
        
        self.enforcement_stats['operations_blocked'] += len(result.blocked_operations)
        
        # Count cross-border transfer violations
        cross_border_violations = [
            v for v in result.violations 
            if v.violation_type == ResidencyViolationType.CROSS_BORDER_TRANSFER
        ]
        self.enforcement_stats['cross_border_transfers_prevented'] += len(cross_border_violations)
        
        # Update average enforcement time
        current_avg = self.enforcement_stats['average_enforcement_time_ms']
        total_checks = self.enforcement_stats['total_checks']
        self.enforcement_stats['average_enforcement_time_ms'] = (
            (current_avg * (total_checks - 1) + result.enforcement_duration_ms) / total_checks
        )
    
    async def _create_residency_tables(self):
        """Create residency enforcement database tables"""
        
        if not self.db_pool:
            return
        
        schema_sql = """
        -- Residency enforcement results
        CREATE TABLE IF NOT EXISTS residency_enforcement_results (
            result_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
            execution_id VARCHAR(100) NOT NULL,
            tenant_id INTEGER NOT NULL,
            
            -- Enforcement results
            is_compliant BOOLEAN NOT NULL,
            violations_count INTEGER NOT NULL DEFAULT 0,
            warnings_count INTEGER NOT NULL DEFAULT 0,
            blocked_operations_count INTEGER NOT NULL DEFAULT 0,
            allowed_operations_count INTEGER NOT NULL DEFAULT 0,
            
            -- Performance
            enforcement_duration_ms FLOAT NOT NULL DEFAULT 0,
            
            -- Context
            workflow_id VARCHAR(100),
            current_infrastructure_region VARCHAR(50),
            
            -- Timestamps
            enforced_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
            
            -- Constraints
            CONSTRAINT chk_residency_counts CHECK (violations_count >= 0 AND warnings_count >= 0)
        );
        
        -- Residency violations
        CREATE TABLE IF NOT EXISTS residency_violations (
            violation_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
            result_id UUID REFERENCES residency_enforcement_results(result_id) ON DELETE CASCADE,
            
            -- Violation details
            policy_id VARCHAR(100) NOT NULL,
            violation_type VARCHAR(50) NOT NULL,
            severity VARCHAR(20) NOT NULL,
            message TEXT NOT NULL,
            
            -- Context
            workflow_id VARCHAR(100) NOT NULL,
            execution_id VARCHAR(100) NOT NULL,
            data_type VARCHAR(100),
            
            -- Geographic details
            source_region VARCHAR(20),
            target_region VARCHAR(20),
            current_infrastructure_region VARCHAR(20),
            
            -- Timestamps
            detected_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
            
            -- Constraints
            CONSTRAINT chk_residency_violation_type CHECK (violation_type IN ('cross_border_transfer', 'unauthorized_region', 'missing_residency_tag', 'invalid_infrastructure')),
            CONSTRAINT chk_residency_severity CHECK (severity IN ('critical', 'high', 'medium', 'low'))
        );
        
        -- Indexes
        CREATE INDEX IF NOT EXISTS idx_residency_enforcement_execution ON residency_enforcement_results(execution_id, enforced_at DESC);
        CREATE INDEX IF NOT EXISTS idx_residency_enforcement_tenant ON residency_enforcement_results(tenant_id, enforced_at DESC);
        CREATE INDEX IF NOT EXISTS idx_residency_enforcement_compliant ON residency_enforcement_results(is_compliant, enforced_at DESC);
        
        CREATE INDEX IF NOT EXISTS idx_residency_violations_result ON residency_violations(result_id);
        CREATE INDEX IF NOT EXISTS idx_residency_violations_policy ON residency_violations(policy_id, severity);
        CREATE INDEX IF NOT EXISTS idx_residency_violations_execution ON residency_violations(execution_id, detected_at DESC);
        CREATE INDEX IF NOT EXISTS idx_residency_violations_type ON residency_violations(violation_type, detected_at DESC);
        """
        
        try:
            async with self.db_pool.acquire() as conn:
                await conn.execute(schema_sql)
            self.logger.info("✅ Residency enforcement tables created")
        except Exception as e:
            self.logger.error(f"❌ Failed to create residency enforcement tables: {e}")
            raise
    
    async def _store_enforcement_result(
        self, result: ResidencyEnforcementResult, tenant_id: int
    ):
        """Store enforcement result in database"""
        
        if not self.db_pool:
            return
        
        try:
            async with self.db_pool.acquire() as conn:
                # Insert enforcement result
                result_query = """
                    INSERT INTO residency_enforcement_results (
                        execution_id, tenant_id, is_compliant, violations_count,
                        warnings_count, blocked_operations_count, allowed_operations_count,
                        enforcement_duration_ms
                    ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8)
                    RETURNING result_id
                """
                
                result_id = await conn.fetchval(
                    result_query,
                    result.execution_id,
                    tenant_id,
                    result.is_compliant,
                    len(result.violations),
                    len(result.warnings),
                    len(result.blocked_operations),
                    len(result.allowed_operations),
                    result.enforcement_duration_ms
                )
                
                # Insert violations
                if result.violations:
                    violation_query = """
                        INSERT INTO residency_violations (
                            result_id, policy_id, violation_type, severity,
                            message, workflow_id, execution_id, data_type,
                            source_region, target_region, current_infrastructure_region,
                            detected_at
                        ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12)
                    """
                    
                    for violation in result.violations:
                        await conn.execute(
                            violation_query,
                            result_id,
                            violation.policy_id,
                            violation.violation_type.value,
                            violation.severity,
                            violation.message,
                            violation.workflow_id,
                            violation.execution_id,
                            violation.data_type,
                            violation.source_region.value if violation.source_region else None,
                            violation.target_region.value if violation.target_region else None,
                            violation.current_infrastructure_region.value if violation.current_infrastructure_region else None,
                            violation.detected_at
                        )
                
        except Exception as e:
            self.logger.error(f"❌ Failed to store residency enforcement result: {e}")
    
    def get_enforcement_statistics(self) -> Dict[str, Any]:
        """Get enforcement statistics"""
        
        compliance_rate = 0.0
        if self.enforcement_stats['total_checks'] > 0:
            compliance_rate = (
                self.enforcement_stats['compliant_operations'] / 
                self.enforcement_stats['total_checks']
            ) * 100
        
        return {
            'total_checks': self.enforcement_stats['total_checks'],
            'compliant_operations': self.enforcement_stats['compliant_operations'],
            'violations_detected': self.enforcement_stats['violations_detected'],
            'operations_blocked': self.enforcement_stats['operations_blocked'],
            'cross_border_transfers_prevented': self.enforcement_stats['cross_border_transfers_prevented'],
            'compliance_rate_percentage': round(compliance_rate, 2),
            'average_enforcement_time_ms': round(self.enforcement_stats['average_enforcement_time_ms'], 2),
            'active_policies_count': len([p for p in self.policy_manager.policies.values() if p.is_active]),
            'supported_regions': [r.value for r in DataResidencyRegion],
            'strict_enforcement_enabled': self.config['strict_enforcement'],
            'fail_closed_enabled': self.config['fail_closed']
        }


# Global residency enforcement engine instance
residency_enforcement_engine = ResidencyEnforcementEngine()