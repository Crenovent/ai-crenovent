"""
SLA Tier Enforcement Engine
Task 9.1.11: Automate SLA tier enforcement in Control Plane

Provides predictable service levels with T0/T1/T2 tier enforcement,
resource allocation, priority queuing, and SLA monitoring.
"""

import asyncio
import json
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict, field
from enum import Enum
import uuid
import time

logger = logging.getLogger(__name__)

class SLATier(Enum):
    """SLA tier definitions with service guarantees"""
    T0_REGULATED = "T0"      # 99.9% uptime, <5s response, dedicated resources
    T1_ENTERPRISE = "T1"     # 99.5% uptime, <15s response, priority resources
    T2_STANDARD = "T2"       # 99.0% uptime, <30s response, shared resources

class ResourceType(Enum):
    """Resource types for SLA enforcement"""
    CPU = "cpu"
    MEMORY = "memory"
    STORAGE = "storage"
    NETWORK = "network"
    CONCURRENT_REQUESTS = "concurrent_requests"
    API_CALLS_PER_MINUTE = "api_calls_per_minute"

class SLAStatus(Enum):
    """SLA compliance status"""
    COMPLIANT = "compliant"
    AT_RISK = "at_risk"
    VIOLATED = "violated"
    DEGRADED = "degraded"

@dataclass
class SLAConfiguration:
    """SLA tier configuration with guarantees and limits"""
    tier: SLATier
    
    # Performance guarantees
    max_response_time_ms: int
    uptime_percentage: float
    availability_slo: float
    
    # Resource allocations
    cpu_cores: float
    memory_gb: float
    storage_gb: float
    concurrent_requests: int
    api_calls_per_minute: int
    
    # Priority and queuing
    priority_weight: int
    queue_timeout_ms: int
    retry_attempts: int
    
    # Monitoring and alerting
    error_budget_percentage: float
    alert_threshold_percentage: float
    escalation_threshold_percentage: float
    
    # Cost and billing
    cost_multiplier: float
    billing_tier: str
    
    def get_resource_limit(self, resource_type: ResourceType) -> float:
        """Get resource limit for specific resource type"""
        resource_map = {
            ResourceType.CPU: self.cpu_cores,
            ResourceType.MEMORY: self.memory_gb,
            ResourceType.STORAGE: self.storage_gb,
            ResourceType.CONCURRENT_REQUESTS: self.concurrent_requests,
            ResourceType.API_CALLS_PER_MINUTE: self.api_calls_per_minute
        }
        return resource_map.get(resource_type, 0.0)

@dataclass
class SLAMetrics:
    """SLA metrics and monitoring data"""
    tenant_id: int
    sla_tier: SLATier
    
    # Performance metrics
    current_response_time_ms: float
    average_response_time_ms: float
    p95_response_time_ms: float
    p99_response_time_ms: float
    
    # Availability metrics
    uptime_percentage: float
    error_rate_percentage: float
    success_rate_percentage: float
    
    # Resource utilization
    cpu_utilization_percentage: float
    memory_utilization_percentage: float
    concurrent_requests_count: int
    api_calls_current_minute: int
    
    # SLA compliance
    sla_status: SLAStatus
    error_budget_remaining_percentage: float
    time_to_sla_violation_minutes: Optional[int]
    
    # Timestamps
    measurement_timestamp: datetime
    last_violation_timestamp: Optional[datetime] = None
    
    def is_compliant(self, config: SLAConfiguration) -> bool:
        """Check if current metrics are SLA compliant"""
        return (
            self.current_response_time_ms <= config.max_response_time_ms and
            self.uptime_percentage >= config.uptime_percentage and
            self.error_rate_percentage <= (100 - config.availability_slo) and
            self.cpu_utilization_percentage <= 80.0 and  # 80% utilization threshold
            self.memory_utilization_percentage <= 80.0 and
            self.concurrent_requests_count <= config.concurrent_requests and
            self.api_calls_current_minute <= config.api_calls_per_minute
        )

@dataclass
class SLAEnforcementAction:
    """SLA enforcement action to be taken"""
    action_id: str
    tenant_id: int
    sla_tier: SLATier
    action_type: str  # throttle, prioritize, scale_up, scale_down, alert
    action_details: Dict[str, Any]
    reason: str
    created_at: datetime
    executed_at: Optional[datetime] = None
    status: str = "pending"  # pending, executed, failed

class SLAEnforcementEngine:
    """
    SLA Tier Enforcement Engine
    Task 9.1.11: Predictable service levels with T0/T1/T2 enforcement
    """
    
    def __init__(self, pool_manager):
        self.pool_manager = pool_manager
        self.logger = logging.getLogger(__name__)
        
        # SLA configurations per tier
        self.sla_configs = {
            SLATier.T0_REGULATED: SLAConfiguration(
                tier=SLATier.T0_REGULATED,
                max_response_time_ms=5000,
                uptime_percentage=99.9,
                availability_slo=99.9,
                cpu_cores=8.0,
                memory_gb=32.0,
                storage_gb=1000.0,
                concurrent_requests=1000,
                api_calls_per_minute=10000,
                priority_weight=100,
                queue_timeout_ms=2000,
                retry_attempts=5,
                error_budget_percentage=0.1,
                alert_threshold_percentage=50.0,
                escalation_threshold_percentage=80.0,
                cost_multiplier=5.0,
                billing_tier="premium"
            ),
            SLATier.T1_ENTERPRISE: SLAConfiguration(
                tier=SLATier.T1_ENTERPRISE,
                max_response_time_ms=15000,
                uptime_percentage=99.5,
                availability_slo=99.5,
                cpu_cores=4.0,
                memory_gb=16.0,
                storage_gb=500.0,
                concurrent_requests=500,
                api_calls_per_minute=5000,
                priority_weight=50,
                queue_timeout_ms=5000,
                retry_attempts=3,
                error_budget_percentage=0.5,
                alert_threshold_percentage=60.0,
                escalation_threshold_percentage=85.0,
                cost_multiplier=2.0,
                billing_tier="enterprise"
            ),
            SLATier.T2_STANDARD: SLAConfiguration(
                tier=SLATier.T2_STANDARD,
                max_response_time_ms=30000,
                uptime_percentage=99.0,
                availability_slo=99.0,
                cpu_cores=2.0,
                memory_gb=8.0,
                storage_gb=100.0,
                concurrent_requests=100,
                api_calls_per_minute=1000,
                priority_weight=10,
                queue_timeout_ms=10000,
                retry_attempts=2,
                error_budget_percentage=1.0,
                alert_threshold_percentage=70.0,
                escalation_threshold_percentage=90.0,
                cost_multiplier=1.0,
                billing_tier="standard"
            )
        }
        
        # Runtime state
        self.tenant_metrics = {}  # tenant_id -> SLAMetrics
        self.enforcement_actions = {}  # action_id -> SLAEnforcementAction
        self.resource_allocations = {}  # tenant_id -> resource allocations
        self.priority_queues = {tier: [] for tier in SLATier}
        
        # Monitoring and alerting
        self.alert_handlers = []
        self.metrics_collectors = []
        
        # Performance tracking
        self.enforcement_stats = {
            "actions_executed": 0,
            "sla_violations": 0,
            "escalations": 0,
            "throttling_events": 0,
            "scaling_events": 0
        }
        
    async def initialize(self) -> bool:
        """Initialize the SLA enforcement engine"""
        try:
            self.logger.info("🚀 Initializing SLA Enforcement Engine...")
            
            # Create database tables
            await self._create_sla_tables()
            
            # Load tenant SLA configurations
            await self._load_tenant_sla_configs()
            
            # Initialize resource monitoring
            await self._initialize_resource_monitoring()
            
            # Start background enforcement loop
            asyncio.create_task(self._enforcement_loop())
            
            self.logger.info("✅ SLA Enforcement Engine initialized successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Failed to initialize SLA Enforcement Engine: {e}")
            return False
    
    async def enforce_sla_for_request(self, 
                                    tenant_id: int, 
                                    request_type: str,
                                    estimated_duration_ms: int = None) -> Dict[str, Any]:
        """
        Enforce SLA for incoming request
        Returns enforcement decision and any actions taken
        """
        try:
            # Get tenant SLA tier
            sla_tier = await self._get_tenant_sla_tier(tenant_id)
            if not sla_tier:
                sla_tier = SLATier.T2_STANDARD  # Default tier
            
            config = self.sla_configs[sla_tier]
            
            # Check current resource utilization
            current_metrics = await self._get_current_metrics(tenant_id, sla_tier)
            
            # Make enforcement decision
            enforcement_decision = await self._make_enforcement_decision(
                tenant_id, sla_tier, config, current_metrics, request_type, estimated_duration_ms
            )
            
            # Execute enforcement actions if needed
            if enforcement_decision["actions"]:
                for action in enforcement_decision["actions"]:
                    await self._execute_enforcement_action(action)
            
            return {
                "tenant_id": tenant_id,
                "sla_tier": sla_tier.value,
                "decision": enforcement_decision["decision"],
                "priority": enforcement_decision["priority"],
                "queue_position": enforcement_decision.get("queue_position"),
                "estimated_wait_ms": enforcement_decision.get("estimated_wait_ms"),
                "resource_allocation": enforcement_decision.get("resource_allocation"),
                "actions_taken": len(enforcement_decision["actions"]),
                "compliance_status": current_metrics.sla_status.value if current_metrics else "unknown"
            }
            
        except Exception as e:
            self.logger.error(f"❌ SLA enforcement failed for tenant {tenant_id}: {e}")
            return {
                "tenant_id": tenant_id,
                "decision": "allow",  # Fail open for availability
                "priority": 10,
                "error": str(e)
            }
    
    async def update_tenant_sla_tier(self, tenant_id: int, new_tier: SLATier, updated_by: str) -> bool:
        """
        Update tenant's SLA tier
        """
        try:
            # Validate tier change
            old_tier = await self._get_tenant_sla_tier(tenant_id)
            
            # Update in database
            if self.pool_manager and hasattr(self.pool_manager, 'postgres_pool'):
                async with self.pool_manager.postgres_pool.acquire() as conn:
                    await conn.execute("""
                        INSERT INTO tenant_sla_configs (tenant_id, sla_tier, updated_by, updated_at)
                        VALUES ($1, $2, $3, NOW())
                        ON CONFLICT (tenant_id) DO UPDATE SET
                        sla_tier = EXCLUDED.sla_tier,
                        updated_by = EXCLUDED.updated_by,
                        updated_at = EXCLUDED.updated_at
                    """, tenant_id, new_tier.value, updated_by)
            
            # Reallocate resources based on new tier
            await self._reallocate_tenant_resources(tenant_id, new_tier)
            
            # Generate evidence pack for tier change
            await self._generate_sla_change_evidence(tenant_id, old_tier, new_tier, updated_by)
            
            self.logger.info(f"✅ Updated SLA tier for tenant {tenant_id}: {old_tier} -> {new_tier}")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Failed to update SLA tier for tenant {tenant_id}: {e}")
            return False
    
    async def get_sla_metrics(self, tenant_id: int) -> Optional[SLAMetrics]:
        """
        Get current SLA metrics for tenant
        """
        try:
            return self.tenant_metrics.get(tenant_id)
        except Exception as e:
            self.logger.error(f"❌ Failed to get SLA metrics for tenant {tenant_id}: {e}")
            return None
    
    async def get_sla_status_report(self, tenant_id: int = None) -> Dict[str, Any]:
        """
        Generate SLA status report for tenant or all tenants
        """
        try:
            if tenant_id:
                # Single tenant report
                metrics = await self.get_sla_metrics(tenant_id)
                if not metrics:
                    return {"error": f"No metrics found for tenant {tenant_id}"}
                
                sla_tier = await self._get_tenant_sla_tier(tenant_id)
                config = self.sla_configs.get(sla_tier, self.sla_configs[SLATier.T2_STANDARD])
                
                return {
                    "tenant_id": tenant_id,
                    "sla_tier": sla_tier.value if sla_tier else "unknown",
                    "metrics": asdict(metrics),
                    "configuration": asdict(config),
                    "compliance": metrics.is_compliant(config),
                    "timestamp": datetime.now().isoformat()
                }
            else:
                # All tenants report
                report = {
                    "total_tenants": len(self.tenant_metrics),
                    "tenants_by_tier": {},
                    "compliance_summary": {
                        "compliant": 0,
                        "at_risk": 0,
                        "violated": 0,
                        "degraded": 0
                    },
                    "enforcement_stats": self.enforcement_stats.copy(),
                    "timestamp": datetime.now().isoformat()
                }
                
                # Count tenants by tier and compliance status
                for tenant_id, metrics in self.tenant_metrics.items():
                    sla_tier = await self._get_tenant_sla_tier(tenant_id)
                    tier_name = sla_tier.value if sla_tier else "unknown"
                    
                    report["tenants_by_tier"][tier_name] = report["tenants_by_tier"].get(tier_name, 0) + 1
                    report["compliance_summary"][metrics.sla_status.value] += 1
                
                return report
                
        except Exception as e:
            self.logger.error(f"❌ Failed to generate SLA status report: {e}")
            return {"error": str(e)}
    
    async def trigger_sla_escalation(self, tenant_id: int, violation_type: str, details: Dict[str, Any]) -> bool:
        """
        Trigger SLA escalation for violations
        """
        try:
            sla_tier = await self._get_tenant_sla_tier(tenant_id)
            config = self.sla_configs.get(sla_tier, self.sla_configs[SLATier.T2_STANDARD])
            
            # Create escalation action
            escalation_action = SLAEnforcementAction(
                action_id=str(uuid.uuid4()),
                tenant_id=tenant_id,
                sla_tier=sla_tier,
                action_type="escalate",
                action_details={
                    "violation_type": violation_type,
                    "details": details,
                    "escalation_level": "sla_violation"
                },
                reason=f"SLA violation: {violation_type}",
                created_at=datetime.now()
            )
            
            # Execute escalation
            await self._execute_enforcement_action(escalation_action)
            
            # Update metrics
            self.enforcement_stats["escalations"] += 1
            self.enforcement_stats["sla_violations"] += 1
            
            self.logger.warning(f"🚨 SLA escalation triggered for tenant {tenant_id}: {violation_type}")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ SLA escalation failed for tenant {tenant_id}: {e}")
            return False
    
    # Private helper methods
    
    async def _create_sla_tables(self):
        """Create database tables for SLA enforcement"""
        if not self.pool_manager or not hasattr(self.pool_manager, 'postgres_pool'):
            return
        
        async with self.pool_manager.postgres_pool.acquire() as conn:
            # Tenant SLA configurations
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS tenant_sla_configs (
                    tenant_id INTEGER PRIMARY KEY,
                    sla_tier VARCHAR(10) NOT NULL,
                    updated_by VARCHAR(255) NOT NULL,
                    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                    
                    CHECK (sla_tier IN ('T0', 'T1', 'T2'))
                )
            """)
            
            # SLA metrics history
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS sla_metrics_history (
                    metric_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                    tenant_id INTEGER NOT NULL,
                    sla_tier VARCHAR(10) NOT NULL,
                    metrics_data JSONB NOT NULL,
                    compliance_status VARCHAR(20) NOT NULL,
                    measurement_timestamp TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                    
                    INDEX(tenant_id),
                    INDEX(measurement_timestamp),
                    INDEX(compliance_status)
                )
            """)
            
            # SLA enforcement actions
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS sla_enforcement_actions (
                    action_id UUID PRIMARY KEY,
                    tenant_id INTEGER NOT NULL,
                    sla_tier VARCHAR(10) NOT NULL,
                    action_type VARCHAR(50) NOT NULL,
                    action_details JSONB NOT NULL,
                    reason TEXT NOT NULL,
                    status VARCHAR(20) NOT NULL DEFAULT 'pending',
                    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                    executed_at TIMESTAMPTZ,
                    
                    INDEX(tenant_id),
                    INDEX(action_type),
                    INDEX(status),
                    INDEX(created_at)
                )
            """)
    
    async def _load_tenant_sla_configs(self):
        """Load tenant SLA configurations from database"""
        if not self.pool_manager or not hasattr(self.pool_manager, 'postgres_pool'):
            return
        
        try:
            async with self.pool_manager.postgres_pool.acquire() as conn:
                results = await conn.fetch("SELECT tenant_id, sla_tier FROM tenant_sla_configs")
                
                for row in results:
                    tenant_id = row['tenant_id']
                    sla_tier = SLATier(row['sla_tier'])
                    
                    # Initialize metrics for tenant
                    if tenant_id not in self.tenant_metrics:
                        self.tenant_metrics[tenant_id] = SLAMetrics(
                            tenant_id=tenant_id,
                            sla_tier=sla_tier,
                            current_response_time_ms=0.0,
                            average_response_time_ms=0.0,
                            p95_response_time_ms=0.0,
                            p99_response_time_ms=0.0,
                            uptime_percentage=100.0,
                            error_rate_percentage=0.0,
                            success_rate_percentage=100.0,
                            cpu_utilization_percentage=0.0,
                            memory_utilization_percentage=0.0,
                            concurrent_requests_count=0,
                            api_calls_current_minute=0,
                            sla_status=SLAStatus.COMPLIANT,
                            error_budget_remaining_percentage=100.0,
                            time_to_sla_violation_minutes=None,
                            measurement_timestamp=datetime.now()
                        )
                    
                    # Allocate resources based on tier
                    await self._allocate_tenant_resources(tenant_id, sla_tier)
                    
        except Exception as e:
            self.logger.error(f"Failed to load tenant SLA configs: {e}")
    
    async def _initialize_resource_monitoring(self):
        """Initialize resource monitoring and metrics collection"""
        # This would integrate with monitoring systems like Prometheus
        pass
    
    async def _enforcement_loop(self):
        """Background loop for continuous SLA enforcement"""
        while True:
            try:
                # Update metrics for all tenants
                await self._update_all_tenant_metrics()
                
                # Check for SLA violations and take actions
                await self._check_sla_violations()
                
                # Process priority queues
                await self._process_priority_queues()
                
                # Clean up old enforcement actions
                await self._cleanup_old_actions()
                
                # Sleep for monitoring interval
                await asyncio.sleep(30)  # 30 second monitoring cycle
                
            except Exception as e:
                self.logger.error(f"SLA enforcement loop error: {e}")
                await asyncio.sleep(60)  # Longer sleep on error
    
    async def _get_tenant_sla_tier(self, tenant_id: int) -> Optional[SLATier]:
        """Get SLA tier for tenant"""
        if self.pool_manager and hasattr(self.pool_manager, 'postgres_pool'):
            try:
                async with self.pool_manager.postgres_pool.acquire() as conn:
                    result = await conn.fetchrow(
                        "SELECT sla_tier FROM tenant_sla_configs WHERE tenant_id = $1",
                        tenant_id
                    )
                    if result:
                        return SLATier(result['sla_tier'])
            except Exception as e:
                self.logger.error(f"Failed to get SLA tier for tenant {tenant_id}: {e}")
        
        return None
    
    async def _get_current_metrics(self, tenant_id: int, sla_tier: SLATier) -> Optional[SLAMetrics]:
        """Get current metrics for tenant"""
        # This would collect real metrics from monitoring systems
        # For now, return cached metrics or create default
        if tenant_id in self.tenant_metrics:
            return self.tenant_metrics[tenant_id]
        
        # Create default metrics
        default_metrics = SLAMetrics(
            tenant_id=tenant_id,
            sla_tier=sla_tier,
            current_response_time_ms=100.0,
            average_response_time_ms=150.0,
            p95_response_time_ms=300.0,
            p99_response_time_ms=500.0,
            uptime_percentage=99.8,
            error_rate_percentage=0.1,
            success_rate_percentage=99.9,
            cpu_utilization_percentage=45.0,
            memory_utilization_percentage=60.0,
            concurrent_requests_count=10,
            api_calls_current_minute=50,
            sla_status=SLAStatus.COMPLIANT,
            error_budget_remaining_percentage=95.0,
            time_to_sla_violation_minutes=None,
            measurement_timestamp=datetime.now()
        )
        
        self.tenant_metrics[tenant_id] = default_metrics
        return default_metrics
    
    async def _make_enforcement_decision(self, 
                                       tenant_id: int, 
                                       sla_tier: SLATier, 
                                       config: SLAConfiguration,
                                       metrics: SLAMetrics,
                                       request_type: str,
                                       estimated_duration_ms: int = None) -> Dict[str, Any]:
        """Make SLA enforcement decision for request"""
        actions = []
        
        # Check if request should be allowed
        if not metrics.is_compliant(config):
            # SLA violation - may need throttling or queuing
            if metrics.concurrent_requests_count >= config.concurrent_requests:
                # Add to priority queue
                queue_position = len(self.priority_queues[sla_tier])
                estimated_wait_ms = queue_position * 1000  # Rough estimate
                
                return {
                    "decision": "queue",
                    "priority": config.priority_weight,
                    "queue_position": queue_position,
                    "estimated_wait_ms": estimated_wait_ms,
                    "actions": actions
                }
            elif metrics.api_calls_current_minute >= config.api_calls_per_minute:
                # Rate limiting
                actions.append(SLAEnforcementAction(
                    action_id=str(uuid.uuid4()),
                    tenant_id=tenant_id,
                    sla_tier=sla_tier,
                    action_type="throttle",
                    action_details={"reason": "rate_limit_exceeded"},
                    reason="API rate limit exceeded",
                    created_at=datetime.now()
                ))
                
                return {
                    "decision": "throttle",
                    "priority": config.priority_weight,
                    "throttle_duration_ms": 60000,  # 1 minute
                    "actions": actions
                }
        
        # Allow request with appropriate priority
        return {
            "decision": "allow",
            "priority": config.priority_weight,
            "resource_allocation": {
                "cpu_cores": config.cpu_cores,
                "memory_gb": config.memory_gb,
                "timeout_ms": config.max_response_time_ms
            },
            "actions": actions
        }
    
    async def _execute_enforcement_action(self, action: SLAEnforcementAction):
        """Execute SLA enforcement action"""
        try:
            if action.action_type == "throttle":
                await self._execute_throttling(action)
            elif action.action_type == "prioritize":
                await self._execute_prioritization(action)
            elif action.action_type == "scale_up":
                await self._execute_scaling(action, scale_up=True)
            elif action.action_type == "scale_down":
                await self._execute_scaling(action, scale_up=False)
            elif action.action_type == "alert":
                await self._execute_alerting(action)
            elif action.action_type == "escalate":
                await self._execute_escalation(action)
            
            # Mark action as executed
            action.executed_at = datetime.now()
            action.status = "executed"
            
            # Store action in database
            await self._store_enforcement_action(action)
            
            self.enforcement_stats["actions_executed"] += 1
            
        except Exception as e:
            action.status = "failed"
            self.logger.error(f"Failed to execute enforcement action {action.action_id}: {e}")
    
    async def _execute_throttling(self, action: SLAEnforcementAction):
        """Execute throttling action"""
        self.enforcement_stats["throttling_events"] += 1
        self.logger.info(f"🚦 Throttling tenant {action.tenant_id}: {action.reason}")
    
    async def _execute_prioritization(self, action: SLAEnforcementAction):
        """Execute prioritization action"""
        self.logger.info(f"⬆️ Prioritizing tenant {action.tenant_id}: {action.reason}")
    
    async def _execute_scaling(self, action: SLAEnforcementAction, scale_up: bool):
        """Execute scaling action"""
        direction = "up" if scale_up else "down"
        self.enforcement_stats["scaling_events"] += 1
        self.logger.info(f"📈 Scaling {direction} for tenant {action.tenant_id}: {action.reason}")
    
    async def _execute_alerting(self, action: SLAEnforcementAction):
        """Execute alerting action"""
        self.logger.warning(f"🚨 SLA alert for tenant {action.tenant_id}: {action.reason}")
    
    async def _execute_escalation(self, action: SLAEnforcementAction):
        """Execute escalation action"""
        self.logger.error(f"🔥 SLA escalation for tenant {action.tenant_id}: {action.reason}")
    
    async def _store_enforcement_action(self, action: SLAEnforcementAction):
        """Store enforcement action in database"""
        if not self.pool_manager or not hasattr(self.pool_manager, 'postgres_pool'):
            return
        
        try:
            async with self.pool_manager.postgres_pool.acquire() as conn:
                await conn.execute("""
                    INSERT INTO sla_enforcement_actions 
                    (action_id, tenant_id, sla_tier, action_type, action_details, reason, status, executed_at)
                    VALUES ($1, $2, $3, $4, $5, $6, $7, $8)
                """, 
                action.action_id,
                action.tenant_id,
                action.sla_tier.value,
                action.action_type,
                json.dumps(action.action_details),
                action.reason,
                action.status,
                action.executed_at)
        except Exception as e:
            self.logger.error(f"Failed to store enforcement action: {e}")
    
    async def _allocate_tenant_resources(self, tenant_id: int, sla_tier: SLATier):
        """Allocate resources based on SLA tier"""
        config = self.sla_configs[sla_tier]
        
        self.resource_allocations[tenant_id] = {
            "cpu_cores": config.cpu_cores,
            "memory_gb": config.memory_gb,
            "storage_gb": config.storage_gb,
            "concurrent_requests": config.concurrent_requests,
            "api_calls_per_minute": config.api_calls_per_minute,
            "priority_weight": config.priority_weight
        }
    
    async def _reallocate_tenant_resources(self, tenant_id: int, new_tier: SLATier):
        """Reallocate resources when SLA tier changes"""
        await self._allocate_tenant_resources(tenant_id, new_tier)
        self.logger.info(f"🔄 Reallocated resources for tenant {tenant_id} to tier {new_tier.value}")
    
    async def _update_all_tenant_metrics(self):
        """Update metrics for all tenants"""
        # This would collect real metrics from monitoring systems
        # For now, simulate some metric updates
        for tenant_id in self.tenant_metrics:
            metrics = self.tenant_metrics[tenant_id]
            metrics.measurement_timestamp = datetime.now()
            
            # Simulate some metric variations
            import random
            metrics.current_response_time_ms += random.uniform(-50, 50)
            metrics.cpu_utilization_percentage += random.uniform(-5, 5)
            metrics.memory_utilization_percentage += random.uniform(-3, 3)
            
            # Keep metrics within reasonable bounds
            metrics.current_response_time_ms = max(50, min(5000, metrics.current_response_time_ms))
            metrics.cpu_utilization_percentage = max(0, min(100, metrics.cpu_utilization_percentage))
            metrics.memory_utilization_percentage = max(0, min(100, metrics.memory_utilization_percentage))
    
    async def _check_sla_violations(self):
        """Check for SLA violations and trigger actions"""
        for tenant_id, metrics in self.tenant_metrics.items():
            sla_tier = await self._get_tenant_sla_tier(tenant_id)
            if not sla_tier:
                continue
            
            config = self.sla_configs[sla_tier]
            
            # Check for violations
            if not metrics.is_compliant(config):
                if metrics.sla_status == SLAStatus.COMPLIANT:
                    # First violation
                    metrics.sla_status = SLAStatus.AT_RISK
                    await self._trigger_sla_alert(tenant_id, "sla_at_risk", metrics)
                elif metrics.sla_status == SLAStatus.AT_RISK:
                    # Escalate to violation
                    metrics.sla_status = SLAStatus.VIOLATED
                    await self.trigger_sla_escalation(tenant_id, "sla_violated", asdict(metrics))
            else:
                # Back to compliant
                if metrics.sla_status != SLAStatus.COMPLIANT:
                    metrics.sla_status = SLAStatus.COMPLIANT
                    self.logger.info(f"✅ Tenant {tenant_id} back to SLA compliance")
    
    async def _trigger_sla_alert(self, tenant_id: int, alert_type: str, metrics: SLAMetrics):
        """Trigger SLA alert"""
        alert_action = SLAEnforcementAction(
            action_id=str(uuid.uuid4()),
            tenant_id=tenant_id,
            sla_tier=metrics.sla_tier,
            action_type="alert",
            action_details={"alert_type": alert_type, "metrics": asdict(metrics)},
            reason=f"SLA alert: {alert_type}",
            created_at=datetime.now()
        )
        
        await self._execute_enforcement_action(alert_action)
    
    async def _process_priority_queues(self):
        """Process priority queues for each SLA tier"""
        for tier in [SLATier.T0_REGULATED, SLATier.T1_ENTERPRISE, SLATier.T2_STANDARD]:
            queue = self.priority_queues[tier]
            if queue:
                # Process highest priority requests first
                # This is a simplified implementation
                processed = min(10, len(queue))  # Process up to 10 requests per cycle
                for _ in range(processed):
                    if queue:
                        request = queue.pop(0)
                        self.logger.debug(f"Processing queued request for tier {tier.value}")
    
    async def _cleanup_old_actions(self):
        """Clean up old enforcement actions"""
        cutoff_time = datetime.now() - timedelta(hours=24)
        
        # Remove old actions from memory
        to_remove = []
        for action_id, action in self.enforcement_actions.items():
            if action.created_at < cutoff_time:
                to_remove.append(action_id)
        
        for action_id in to_remove:
            del self.enforcement_actions[action_id]
    
    async def _generate_sla_change_evidence(self, tenant_id: int, old_tier: SLATier, new_tier: SLATier, updated_by: str):
        """Generate evidence pack for SLA tier changes"""
        try:
            # This would integrate with the evidence pack service
            evidence_data = {
                "tenant_id": tenant_id,
                "old_sla_tier": old_tier.value if old_tier else None,
                "new_sla_tier": new_tier.value,
                "updated_by": updated_by,
                "timestamp": datetime.now().isoformat(),
                "resource_changes": {
                    "old_allocation": self.sla_configs[old_tier].__dict__ if old_tier else None,
                    "new_allocation": self.sla_configs[new_tier].__dict__
                }
            }
            
            self.logger.info(f"📋 Generated SLA change evidence for tenant {tenant_id}")
            
        except Exception as e:
            self.logger.error(f"Failed to generate SLA change evidence: {e}")
