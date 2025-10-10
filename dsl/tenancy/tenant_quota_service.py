"""
Task 8.1-T24: Implement tenant quotas (policies applied per tenant limits)
Multi-tenant quota management and enforcement system
"""

import asyncio
import json
import logging
import uuid
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Any, Optional, Set
from dataclasses import dataclass, field
from enum import Enum

import asyncpg

logger = logging.getLogger(__name__)


class QuotaType(Enum):
    """Types of tenant quotas"""
    WORKFLOW_EXECUTIONS = "workflow_executions"
    API_CALLS = "api_calls"
    DATA_PROCESSING_GB = "data_processing_gb"
    STORAGE_GB = "storage_gb"
    CONCURRENT_EXECUTIONS = "concurrent_executions"
    POLICY_EVALUATIONS = "policy_evaluations"
    AUDIT_EVENTS = "audit_events"
    EVIDENCE_PACKS = "evidence_packs"


class QuotaPeriod(Enum):
    """Quota reset periods"""
    HOURLY = "hourly"
    DAILY = "daily"
    WEEKLY = "weekly"
    MONTHLY = "monthly"
    YEARLY = "yearly"


class TenantTier(Enum):
    """Tenant service tiers"""
    FREE = "free"
    STARTER = "starter"
    PROFESSIONAL = "professional"
    ENTERPRISE = "enterprise"
    REGULATED = "regulated"


@dataclass
class TenantQuota:
    """Tenant quota definition"""
    quota_id: str
    tenant_id: int
    quota_type: QuotaType
    
    # Quota limits
    limit_value: int
    current_usage: int = 0
    
    # Time period
    period: QuotaPeriod = QuotaPeriod.MONTHLY
    period_start: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    period_end: datetime = field(default_factory=lambda: datetime.now(timezone.utc) + timedelta(days=30))
    
    # Enforcement settings
    hard_limit: bool = True
    warning_threshold_percentage: float = 80.0
    
    # Metadata
    tier: TenantTier = TenantTier.PROFESSIONAL
    description: str = ""
    is_active: bool = True
    
    # Timestamps
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "quota_id": self.quota_id,
            "tenant_id": self.tenant_id,
            "quota_type": self.quota_type.value,
            "limit_value": self.limit_value,
            "current_usage": self.current_usage,
            "period": self.period.value,
            "period_start": self.period_start.isoformat(),
            "period_end": self.period_end.isoformat(),
            "hard_limit": self.hard_limit,
            "warning_threshold_percentage": self.warning_threshold_percentage,
            "tier": self.tier.value,
            "description": self.description,
            "is_active": self.is_active,
            "usage_percentage": (self.current_usage / self.limit_value * 100) if self.limit_value > 0 else 0,
            "remaining": max(0, self.limit_value - self.current_usage),
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat()
        }


@dataclass
class QuotaViolation:
    """Quota violation event"""
    violation_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    quota_id: str = ""
    tenant_id: int = 0
    
    # Violation details
    quota_type: QuotaType = QuotaType.WORKFLOW_EXECUTIONS
    violation_type: str = "limit_exceeded"
    
    # Usage details
    attempted_usage: int = 0
    current_usage: int = 0
    quota_limit: int = 0
    
    # Context
    workflow_id: Optional[str] = None
    execution_id: Optional[str] = None
    user_id: Optional[str] = None
    
    # Action taken
    action_taken: str = "blocked"  # blocked, warned, allowed
    
    # Timestamps
    occurred_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "violation_id": self.violation_id,
            "quota_id": self.quota_id,
            "tenant_id": self.tenant_id,
            "quota_type": self.quota_type.value,
            "violation_type": self.violation_type,
            "attempted_usage": self.attempted_usage,
            "current_usage": self.current_usage,
            "quota_limit": self.quota_limit,
            "workflow_id": self.workflow_id,
            "execution_id": self.execution_id,
            "user_id": self.user_id,
            "action_taken": self.action_taken,
            "occurred_at": self.occurred_at.isoformat()
        }


@dataclass
class QuotaCheckResult:
    """Result of quota check"""
    tenant_id: int
    quota_type: QuotaType
    is_allowed: bool = True
    
    # Quota details
    current_usage: int = 0
    quota_limit: int = 0
    remaining: int = 0
    usage_percentage: float = 0.0
    
    # Violations
    violations: List[QuotaViolation] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    
    # Performance
    check_duration_ms: float = 0.0
    
    # Timestamps
    checked_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "tenant_id": self.tenant_id,
            "quota_type": self.quota_type.value,
            "is_allowed": self.is_allowed,
            "current_usage": self.current_usage,
            "quota_limit": self.quota_limit,
            "remaining": self.remaining,
            "usage_percentage": self.usage_percentage,
            "violations": [v.to_dict() for v in self.violations],
            "warnings": self.warnings,
            "check_duration_ms": self.check_duration_ms,
            "checked_at": self.checked_at.isoformat()
        }


class TenantQuotaService:
    """
    Tenant Quota Service - Task 8.1-T24
    
    Manages and enforces per-tenant quotas to prevent noisy neighbors
    Provides multi-tier quota management with real-time enforcement
    """
    
    def __init__(self, db_pool: Optional[asyncpg.Pool] = None):
        self.db_pool = db_pool
        self.logger = logging.getLogger(__name__)
        
        # Configuration
        self.config = {
            'enable_quota_enforcement': True,
            'strict_enforcement': True,
            'enable_warnings': True,
            'enable_auto_reset': True,
            'cache_quotas': True,
            'cache_ttl_minutes': 5
        }
        
        # Quota cache
        self.quota_cache: Dict[str, TenantQuota] = {}
        self.cache_last_updated: Optional[datetime] = None
        
        # Default quota configurations by tier
        self.default_quotas = self._initialize_default_quotas()
        
        # Statistics
        self.quota_stats = {
            'total_checks': 0,
            'checks_allowed': 0,
            'checks_blocked': 0,
            'violations_detected': 0,
            'quotas_reset': 0,
            'average_check_time_ms': 0.0
        }
    
    async def initialize(self) -> bool:
        """Initialize tenant quota service"""
        try:
            await self._create_quota_tables()
            await self._load_tenant_quotas()
            self.logger.info("✅ Tenant quota service initialized")
            return True
        except Exception as e:
            self.logger.error(f"❌ Failed to initialize tenant quota service: {e}")
            return False
    
    async def check_quota(
        self,
        tenant_id: int,
        quota_type: QuotaType,
        requested_amount: int = 1,
        context: Optional[Dict[str, Any]] = None
    ) -> QuotaCheckResult:
        """
        Check if tenant has quota available for requested operation
        
        This is the main entry point for quota enforcement
        """
        
        start_time = datetime.now(timezone.utc)
        
        # Initialize result
        result = QuotaCheckResult(
            tenant_id=tenant_id,
            quota_type=quota_type
        )
        
        try:
            # Get tenant quota
            quota = await self._get_tenant_quota(tenant_id, quota_type)
            
            if not quota:
                # Create default quota if none exists
                quota = await self._create_default_quota(tenant_id, quota_type)
            
            # Check if quota period needs reset
            if await self._should_reset_quota(quota):
                await self._reset_quota(quota)
            
            # Update result with current quota state
            result.current_usage = quota.current_usage
            result.quota_limit = quota.limit_value
            result.remaining = max(0, quota.limit_value - quota.current_usage)
            result.usage_percentage = (quota.current_usage / quota.limit_value * 100) if quota.limit_value > 0 else 0
            
            # Check if request would exceed quota
            new_usage = quota.current_usage + requested_amount
            
            if new_usage > quota.limit_value:
                # Quota exceeded
                result.is_allowed = False
                
                violation = QuotaViolation(
                    quota_id=quota.quota_id,
                    tenant_id=tenant_id,
                    quota_type=quota_type,
                    violation_type="limit_exceeded",
                    attempted_usage=requested_amount,
                    current_usage=quota.current_usage,
                    quota_limit=quota.limit_value,
                    workflow_id=context.get('workflow_id') if context else None,
                    execution_id=context.get('execution_id') if context else None,
                    user_id=context.get('user_id') if context else None,
                    action_taken="blocked" if quota.hard_limit else "warned"
                )
                result.violations.append(violation)
                
                # Store violation
                if self.db_pool:
                    await self._store_quota_violation(violation)
                
                if quota.hard_limit:
                    self.logger.warning(f"🚫 Quota exceeded for tenant {tenant_id}: {quota_type.value} ({new_usage}/{quota.limit_value})")
                else:
                    result.is_allowed = True  # Soft limit - allow but warn
                    result.warnings.append(f"Quota soft limit exceeded: {new_usage}/{quota.limit_value}")
                    self.logger.warning(f"⚠️ Quota soft limit exceeded for tenant {tenant_id}: {quota_type.value}")
            
            elif result.usage_percentage >= quota.warning_threshold_percentage:
                # Approaching quota limit
                result.warnings.append(f"Quota usage at {result.usage_percentage:.1f}% of limit")
                self.logger.info(f"⚠️ Quota warning for tenant {tenant_id}: {quota_type.value} at {result.usage_percentage:.1f}%")
            
            # If allowed, increment usage
            if result.is_allowed and self.config['enable_quota_enforcement']:
                await self._increment_quota_usage(quota, requested_amount)
                result.current_usage = quota.current_usage + requested_amount
                result.remaining = max(0, quota.limit_value - result.current_usage)
            
            # Calculate check duration
            end_time = datetime.now(timezone.utc)
            result.check_duration_ms = (end_time - start_time).total_seconds() * 1000
            
            # Update statistics
            self._update_quota_stats(result)
            
            return result
            
        except Exception as e:
            self.logger.error(f"❌ Quota check error for tenant {tenant_id}: {e}")
            
            # Fail open for quota check errors (allow operation but log error)
            result.is_allowed = True
            result.warnings.append(f"Quota check system error: {str(e)}")
            result.check_duration_ms = (datetime.now(timezone.utc) - start_time).total_seconds() * 1000
            
            return result
    
    async def get_tenant_quota_summary(self, tenant_id: int) -> Dict[str, Any]:
        """Get comprehensive quota summary for tenant"""
        
        try:
            quotas = await self._get_all_tenant_quotas(tenant_id)
            
            summary = {
                'tenant_id': tenant_id,
                'total_quotas': len(quotas),
                'quotas': {},
                'overall_usage_percentage': 0.0,
                'quotas_at_warning': 0,
                'quotas_exceeded': 0,
                'next_reset': None
            }
            
            total_usage_percentage = 0.0
            
            for quota in quotas:
                usage_percentage = (quota.current_usage / quota.limit_value * 100) if quota.limit_value > 0 else 0
                total_usage_percentage += usage_percentage
                
                summary['quotas'][quota.quota_type.value] = {
                    'current_usage': quota.current_usage,
                    'limit': quota.limit_value,
                    'remaining': max(0, quota.limit_value - quota.current_usage),
                    'usage_percentage': usage_percentage,
                    'period': quota.period.value,
                    'period_end': quota.period_end.isoformat(),
                    'hard_limit': quota.hard_limit,
                    'tier': quota.tier.value
                }
                
                if usage_percentage >= quota.warning_threshold_percentage:
                    summary['quotas_at_warning'] += 1
                
                if quota.current_usage >= quota.limit_value:
                    summary['quotas_exceeded'] += 1
                
                # Find next reset time
                if not summary['next_reset'] or quota.period_end < datetime.fromisoformat(summary['next_reset']):
                    summary['next_reset'] = quota.period_end.isoformat()
            
            summary['overall_usage_percentage'] = total_usage_percentage / len(quotas) if quotas else 0.0
            
            return summary
            
        except Exception as e:
            self.logger.error(f"❌ Failed to get quota summary for tenant {tenant_id}: {e}")
            return {'error': str(e)}
    
    async def reset_tenant_quotas(self, tenant_id: int, quota_types: List[QuotaType] = None) -> Dict[str, Any]:
        """Reset quotas for tenant (admin operation)"""
        
        try:
            if quota_types:
                quotas_to_reset = []
                for quota_type in quota_types:
                    quota = await self._get_tenant_quota(tenant_id, quota_type)
                    if quota:
                        quotas_to_reset.append(quota)
            else:
                quotas_to_reset = await self._get_all_tenant_quotas(tenant_id)
            
            reset_count = 0
            for quota in quotas_to_reset:
                await self._reset_quota(quota)
                reset_count += 1
            
            self.logger.info(f"✅ Reset {reset_count} quotas for tenant {tenant_id}")
            
            return {
                'tenant_id': tenant_id,
                'quotas_reset': reset_count,
                'reset_at': datetime.now(timezone.utc).isoformat()
            }
            
        except Exception as e:
            self.logger.error(f"❌ Failed to reset quotas for tenant {tenant_id}: {e}")
            return {'error': str(e)}
    
    def _initialize_default_quotas(self) -> Dict[TenantTier, Dict[QuotaType, int]]:
        """Initialize default quota limits by tier"""
        
        return {
            TenantTier.FREE: {
                QuotaType.WORKFLOW_EXECUTIONS: 100,
                QuotaType.API_CALLS: 1000,
                QuotaType.DATA_PROCESSING_GB: 1,
                QuotaType.STORAGE_GB: 1,
                QuotaType.CONCURRENT_EXECUTIONS: 2,
                QuotaType.POLICY_EVALUATIONS: 500,
                QuotaType.AUDIT_EVENTS: 1000,
                QuotaType.EVIDENCE_PACKS: 10
            },
            TenantTier.STARTER: {
                QuotaType.WORKFLOW_EXECUTIONS: 1000,
                QuotaType.API_CALLS: 10000,
                QuotaType.DATA_PROCESSING_GB: 10,
                QuotaType.STORAGE_GB: 10,
                QuotaType.CONCURRENT_EXECUTIONS: 5,
                QuotaType.POLICY_EVALUATIONS: 5000,
                QuotaType.AUDIT_EVENTS: 10000,
                QuotaType.EVIDENCE_PACKS: 100
            },
            TenantTier.PROFESSIONAL: {
                QuotaType.WORKFLOW_EXECUTIONS: 10000,
                QuotaType.API_CALLS: 100000,
                QuotaType.DATA_PROCESSING_GB: 100,
                QuotaType.STORAGE_GB: 100,
                QuotaType.CONCURRENT_EXECUTIONS: 20,
                QuotaType.POLICY_EVALUATIONS: 50000,
                QuotaType.AUDIT_EVENTS: 100000,
                QuotaType.EVIDENCE_PACKS: 1000
            },
            TenantTier.ENTERPRISE: {
                QuotaType.WORKFLOW_EXECUTIONS: 100000,
                QuotaType.API_CALLS: 1000000,
                QuotaType.DATA_PROCESSING_GB: 1000,
                QuotaType.STORAGE_GB: 1000,
                QuotaType.CONCURRENT_EXECUTIONS: 100,
                QuotaType.POLICY_EVALUATIONS: 500000,
                QuotaType.AUDIT_EVENTS: 1000000,
                QuotaType.EVIDENCE_PACKS: 10000
            },
            TenantTier.REGULATED: {
                QuotaType.WORKFLOW_EXECUTIONS: 1000000,
                QuotaType.API_CALLS: 10000000,
                QuotaType.DATA_PROCESSING_GB: 10000,
                QuotaType.STORAGE_GB: 10000,
                QuotaType.CONCURRENT_EXECUTIONS: 500,
                QuotaType.POLICY_EVALUATIONS: 5000000,
                QuotaType.AUDIT_EVENTS: 10000000,
                QuotaType.EVIDENCE_PACKS: 100000
            }
        }
    
    async def _get_tenant_quota(self, tenant_id: int, quota_type: QuotaType) -> Optional[TenantQuota]:
        """Get tenant quota from cache or database"""
        
        cache_key = f"{tenant_id}_{quota_type.value}"
        
        # Check cache first
        if self.config['cache_quotas'] and cache_key in self.quota_cache:
            if self._is_cache_valid():
                return self.quota_cache[cache_key]
        
        # Load from database
        if self.db_pool:
            try:
                async with self.db_pool.acquire() as conn:
                    row = await conn.fetchrow("""
                        SELECT quota_id, tenant_id, quota_type, limit_value, current_usage,
                               period, period_start, period_end, hard_limit, warning_threshold_percentage,
                               tier, description, is_active, created_at, updated_at
                        FROM tenant_quotas
                        WHERE tenant_id = $1 AND quota_type = $2 AND is_active = TRUE
                    """, tenant_id, quota_type.value)
                    
                    if row:
                        quota = TenantQuota(
                            quota_id=row['quota_id'],
                            tenant_id=row['tenant_id'],
                            quota_type=QuotaType(row['quota_type']),
                            limit_value=row['limit_value'],
                            current_usage=row['current_usage'],
                            period=QuotaPeriod(row['period']),
                            period_start=row['period_start'],
                            period_end=row['period_end'],
                            hard_limit=row['hard_limit'],
                            warning_threshold_percentage=row['warning_threshold_percentage'],
                            tier=TenantTier(row['tier']),
                            description=row['description'],
                            is_active=row['is_active'],
                            created_at=row['created_at'],
                            updated_at=row['updated_at']
                        )
                        
                        # Cache the quota
                        if self.config['cache_quotas']:
                            self.quota_cache[cache_key] = quota
                        
                        return quota
            except Exception as e:
                self.logger.error(f"❌ Failed to load quota from database: {e}")
        
        return None
    
    async def _create_default_quota(self, tenant_id: int, quota_type: QuotaType) -> TenantQuota:
        """Create default quota for tenant"""
        
        # Determine tenant tier (would normally come from tenant metadata)
        tenant_tier = TenantTier.PROFESSIONAL  # Default tier
        
        # Get default limit for this tier and quota type
        default_limit = self.default_quotas[tenant_tier].get(quota_type, 1000)
        
        # Calculate period end based on quota type
        if quota_type in [QuotaType.API_CALLS, QuotaType.CONCURRENT_EXECUTIONS]:
            period = QuotaPeriod.HOURLY
            period_end = datetime.now(timezone.utc) + timedelta(hours=1)
        elif quota_type in [QuotaType.WORKFLOW_EXECUTIONS, QuotaType.POLICY_EVALUATIONS]:
            period = QuotaPeriod.DAILY
            period_end = datetime.now(timezone.utc) + timedelta(days=1)
        else:
            period = QuotaPeriod.MONTHLY
            period_end = datetime.now(timezone.utc) + timedelta(days=30)
        
        quota = TenantQuota(
            quota_id=str(uuid.uuid4()),
            tenant_id=tenant_id,
            quota_type=quota_type,
            limit_value=default_limit,
            period=period,
            period_end=period_end,
            tier=tenant_tier,
            description=f"Default {quota_type.value} quota for {tenant_tier.value} tier"
        )
        
        # Store in database
        if self.db_pool:
            await self._store_quota(quota)
        
        # Cache the quota
        if self.config['cache_quotas']:
            cache_key = f"{tenant_id}_{quota_type.value}"
            self.quota_cache[cache_key] = quota
        
        self.logger.info(f"✅ Created default quota for tenant {tenant_id}: {quota_type.value} = {default_limit}")
        
        return quota
    
    async def _should_reset_quota(self, quota: TenantQuota) -> bool:
        """Check if quota period has expired and needs reset"""
        
        return datetime.now(timezone.utc) >= quota.period_end
    
    async def _reset_quota(self, quota: TenantQuota):
        """Reset quota usage and update period"""
        
        quota.current_usage = 0
        quota.updated_at = datetime.now(timezone.utc)
        
        # Calculate new period end
        if quota.period == QuotaPeriod.HOURLY:
            quota.period_start = datetime.now(timezone.utc)
            quota.period_end = quota.period_start + timedelta(hours=1)
        elif quota.period == QuotaPeriod.DAILY:
            quota.period_start = datetime.now(timezone.utc)
            quota.period_end = quota.period_start + timedelta(days=1)
        elif quota.period == QuotaPeriod.WEEKLY:
            quota.period_start = datetime.now(timezone.utc)
            quota.period_end = quota.period_start + timedelta(weeks=1)
        elif quota.period == QuotaPeriod.MONTHLY:
            quota.period_start = datetime.now(timezone.utc)
            quota.period_end = quota.period_start + timedelta(days=30)
        elif quota.period == QuotaPeriod.YEARLY:
            quota.period_start = datetime.now(timezone.utc)
            quota.period_end = quota.period_start + timedelta(days=365)
        
        # Update in database
        if self.db_pool:
            await self._update_quota_in_db(quota)
        
        # Update cache
        if self.config['cache_quotas']:
            cache_key = f"{quota.tenant_id}_{quota.quota_type.value}"
            self.quota_cache[cache_key] = quota
        
        self.quota_stats['quotas_reset'] += 1
        self.logger.info(f"🔄 Reset quota for tenant {quota.tenant_id}: {quota.quota_type.value}")
    
    async def _increment_quota_usage(self, quota: TenantQuota, amount: int):
        """Increment quota usage"""
        
        quota.current_usage += amount
        quota.updated_at = datetime.now(timezone.utc)
        
        # Update in database
        if self.db_pool:
            try:
                async with self.db_pool.acquire() as conn:
                    await conn.execute("""
                        UPDATE tenant_quotas
                        SET current_usage = current_usage + $1, updated_at = NOW()
                        WHERE quota_id = $2
                    """, amount, quota.quota_id)
            except Exception as e:
                self.logger.error(f"❌ Failed to update quota usage in database: {e}")
        
        # Update cache
        if self.config['cache_quotas']:
            cache_key = f"{quota.tenant_id}_{quota.quota_type.value}"
            self.quota_cache[cache_key] = quota
    
    def _update_quota_stats(self, result: QuotaCheckResult):
        """Update quota statistics"""
        
        self.quota_stats['total_checks'] += 1
        
        if result.is_allowed:
            self.quota_stats['checks_allowed'] += 1
        else:
            self.quota_stats['checks_blocked'] += 1
        
        self.quota_stats['violations_detected'] += len(result.violations)
        
        # Update average check time
        current_avg = self.quota_stats['average_check_time_ms']
        total_checks = self.quota_stats['total_checks']
        self.quota_stats['average_check_time_ms'] = (
            (current_avg * (total_checks - 1) + result.check_duration_ms) / total_checks
        )
    
    def _is_cache_valid(self) -> bool:
        """Check if quota cache is still valid"""
        
        if not self.cache_last_updated:
            return False
        
        cache_age = datetime.now(timezone.utc) - self.cache_last_updated
        return cache_age.total_seconds() < (self.config['cache_ttl_minutes'] * 60)
    
    async def _get_all_tenant_quotas(self, tenant_id: int) -> List[TenantQuota]:
        """Get all quotas for tenant"""
        
        quotas = []
        
        if self.db_pool:
            try:
                async with self.db_pool.acquire() as conn:
                    rows = await conn.fetch("""
                        SELECT quota_id, tenant_id, quota_type, limit_value, current_usage,
                               period, period_start, period_end, hard_limit, warning_threshold_percentage,
                               tier, description, is_active, created_at, updated_at
                        FROM tenant_quotas
                        WHERE tenant_id = $1 AND is_active = TRUE
                        ORDER BY quota_type
                    """, tenant_id)
                    
                    for row in rows:
                        quota = TenantQuota(
                            quota_id=row['quota_id'],
                            tenant_id=row['tenant_id'],
                            quota_type=QuotaType(row['quota_type']),
                            limit_value=row['limit_value'],
                            current_usage=row['current_usage'],
                            period=QuotaPeriod(row['period']),
                            period_start=row['period_start'],
                            period_end=row['period_end'],
                            hard_limit=row['hard_limit'],
                            warning_threshold_percentage=row['warning_threshold_percentage'],
                            tier=TenantTier(row['tier']),
                            description=row['description'],
                            is_active=row['is_active'],
                            created_at=row['created_at'],
                            updated_at=row['updated_at']
                        )
                        quotas.append(quota)
            except Exception as e:
                self.logger.error(f"❌ Failed to load tenant quotas: {e}")
        
        return quotas
    
    async def _create_quota_tables(self):
        """Create quota management database tables"""
        
        if not self.db_pool:
            return
        
        schema_sql = """
        -- Tenant quotas
        CREATE TABLE IF NOT EXISTS tenant_quotas (
            quota_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
            tenant_id INTEGER NOT NULL,
            
            -- Quota configuration
            quota_type VARCHAR(50) NOT NULL,
            limit_value INTEGER NOT NULL,
            current_usage INTEGER NOT NULL DEFAULT 0,
            
            -- Time period
            period VARCHAR(20) NOT NULL DEFAULT 'monthly',
            period_start TIMESTAMPTZ NOT NULL DEFAULT NOW(),
            period_end TIMESTAMPTZ NOT NULL,
            
            -- Enforcement settings
            hard_limit BOOLEAN NOT NULL DEFAULT TRUE,
            warning_threshold_percentage FLOAT NOT NULL DEFAULT 80.0,
            
            -- Metadata
            tier VARCHAR(20) NOT NULL DEFAULT 'professional',
            description TEXT,
            is_active BOOLEAN NOT NULL DEFAULT TRUE,
            
            -- Timestamps
            created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
            updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
            
            -- Constraints
            CONSTRAINT chk_quota_type CHECK (quota_type IN ('workflow_executions', 'api_calls', 'data_processing_gb', 'storage_gb', 'concurrent_executions', 'policy_evaluations', 'audit_events', 'evidence_packs')),
            CONSTRAINT chk_quota_period CHECK (period IN ('hourly', 'daily', 'weekly', 'monthly', 'yearly')),
            CONSTRAINT chk_quota_tier CHECK (tier IN ('free', 'starter', 'professional', 'enterprise', 'regulated')),
            CONSTRAINT chk_quota_values CHECK (limit_value >= 0 AND current_usage >= 0),
            CONSTRAINT chk_quota_threshold CHECK (warning_threshold_percentage >= 0 AND warning_threshold_percentage <= 100),
            CONSTRAINT uq_tenant_quota_type UNIQUE (tenant_id, quota_type)
        );
        
        -- Quota violations
        CREATE TABLE IF NOT EXISTS tenant_quota_violations (
            violation_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
            quota_id UUID REFERENCES tenant_quotas(quota_id) ON DELETE CASCADE,
            tenant_id INTEGER NOT NULL,
            
            -- Violation details
            quota_type VARCHAR(50) NOT NULL,
            violation_type VARCHAR(50) NOT NULL DEFAULT 'limit_exceeded',
            
            -- Usage details
            attempted_usage INTEGER NOT NULL,
            current_usage INTEGER NOT NULL,
            quota_limit INTEGER NOT NULL,
            
            -- Context
            workflow_id VARCHAR(100),
            execution_id VARCHAR(100),
            user_id VARCHAR(100),
            
            -- Action taken
            action_taken VARCHAR(20) NOT NULL DEFAULT 'blocked',
            
            -- Timestamps
            occurred_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
            
            -- Constraints
            CONSTRAINT chk_quota_violation_action CHECK (action_taken IN ('blocked', 'warned', 'allowed'))
        );
        
        -- Indexes
        CREATE INDEX IF NOT EXISTS idx_tenant_quotas_tenant ON tenant_quotas(tenant_id, is_active);
        CREATE INDEX IF NOT EXISTS idx_tenant_quotas_type ON tenant_quotas(quota_type, tenant_id);
        CREATE INDEX IF NOT EXISTS idx_tenant_quotas_period_end ON tenant_quotas(period_end) WHERE is_active = TRUE;
        CREATE INDEX IF NOT EXISTS idx_tenant_quotas_usage ON tenant_quotas(tenant_id, current_usage, limit_value);
        
        CREATE INDEX IF NOT EXISTS idx_quota_violations_quota ON tenant_quota_violations(quota_id, occurred_at DESC);
        CREATE INDEX IF NOT EXISTS idx_quota_violations_tenant ON tenant_quota_violations(tenant_id, occurred_at DESC);
        CREATE INDEX IF NOT EXISTS idx_quota_violations_type ON tenant_quota_violations(quota_type, violation_type);
        """
        
        try:
            async with self.db_pool.acquire() as conn:
                await conn.execute(schema_sql)
            self.logger.info("✅ Tenant quota tables created")
        except Exception as e:
            self.logger.error(f"❌ Failed to create quota tables: {e}")
            raise
    
    async def _load_tenant_quotas(self):
        """Load tenant quotas into cache"""
        
        if not self.config['cache_quotas']:
            return
        
        try:
            # This would load frequently accessed quotas into cache
            # For now, we'll populate cache on-demand
            self.cache_last_updated = datetime.now(timezone.utc)
        except Exception as e:
            self.logger.error(f"❌ Failed to load tenant quotas: {e}")
    
    async def _store_quota(self, quota: TenantQuota):
        """Store quota in database"""
        
        if not self.db_pool:
            return
        
        try:
            async with self.db_pool.acquire() as conn:
                await conn.execute("""
                    INSERT INTO tenant_quotas (
                        quota_id, tenant_id, quota_type, limit_value, current_usage,
                        period, period_start, period_end, hard_limit, warning_threshold_percentage,
                        tier, description, is_active
                    ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13)
                """,
                quota.quota_id, quota.tenant_id, quota.quota_type.value, quota.limit_value,
                quota.current_usage, quota.period.value, quota.period_start, quota.period_end,
                quota.hard_limit, quota.warning_threshold_percentage, quota.tier.value,
                quota.description, quota.is_active)
        except Exception as e:
            self.logger.error(f"❌ Failed to store quota in database: {e}")
    
    async def _update_quota_in_db(self, quota: TenantQuota):
        """Update quota in database"""
        
        if not self.db_pool:
            return
        
        try:
            async with self.db_pool.acquire() as conn:
                await conn.execute("""
                    UPDATE tenant_quotas
                    SET current_usage = $1, period_start = $2, period_end = $3, updated_at = NOW()
                    WHERE quota_id = $4
                """, quota.current_usage, quota.period_start, quota.period_end, quota.quota_id)
        except Exception as e:
            self.logger.error(f"❌ Failed to update quota in database: {e}")
    
    async def _store_quota_violation(self, violation: QuotaViolation):
        """Store quota violation in database"""
        
        if not self.db_pool:
            return
        
        try:
            async with self.db_pool.acquire() as conn:
                await conn.execute("""
                    INSERT INTO tenant_quota_violations (
                        violation_id, quota_id, tenant_id, quota_type, violation_type,
                        attempted_usage, current_usage, quota_limit, workflow_id,
                        execution_id, user_id, action_taken, occurred_at
                    ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13)
                """,
                violation.violation_id, violation.quota_id, violation.tenant_id,
                violation.quota_type.value, violation.violation_type, violation.attempted_usage,
                violation.current_usage, violation.quota_limit, violation.workflow_id,
                violation.execution_id, violation.user_id, violation.action_taken,
                violation.occurred_at)
        except Exception as e:
            self.logger.error(f"❌ Failed to store quota violation: {e}")
    
    def get_quota_statistics(self) -> Dict[str, Any]:
        """Get quota service statistics"""
        
        success_rate = 0.0
        if self.quota_stats['total_checks'] > 0:
            success_rate = (
                self.quota_stats['checks_allowed'] / 
                self.quota_stats['total_checks']
            ) * 100
        
        return {
            'total_checks': self.quota_stats['total_checks'],
            'checks_allowed': self.quota_stats['checks_allowed'],
            'checks_blocked': self.quota_stats['checks_blocked'],
            'violations_detected': self.quota_stats['violations_detected'],
            'quotas_reset': self.quota_stats['quotas_reset'],
            'success_rate_percentage': round(success_rate, 2),
            'average_check_time_ms': round(self.quota_stats['average_check_time_ms'], 2),
            'cached_quotas_count': len(self.quota_cache),
            'supported_quota_types': [qt.value for qt in QuotaType],
            'supported_tiers': [tt.value for tt in TenantTier],
            'quota_enforcement_enabled': self.config['enable_quota_enforcement'],
            'cache_enabled': self.config['cache_quotas']
        }


# Global tenant quota service instance
tenant_quota_service = TenantQuotaService()
