"""
API Consumption Contracts API
============================
Implements Chapter 11.4 tasks for API contract management and governance

Features:
- Task 11.4.3: Build API registry and discovery catalog
- Task 11.4.4: Define SLA tiers (T0/T1/T2) in API contracts
- Task 11.4.17: Configure retry/backoff + idempotency for APIs
- Task 11.4.18: Add rate limiting and quotas per API key
- Task 11.4.20: Configure FinOps cost attribution per API
- Task 11.4.21: Create budget alerts for API usage
- Task 11.4.23: Create API documentation generator
- Task 11.4.27: Configure SLA-aware monitoring on APIs
- Task 11.4.28: Build persona dashboards for API usage
"""

from fastapi import APIRouter, HTTPException, Depends, Query, Header
from typing import Dict, List, Any, Optional
from pydantic import BaseModel, Field
import logging
from datetime import datetime, timezone, timedelta
import uuid
import asyncio
from enum import Enum

from dsl.governance.audit_trail_service import AuditTrailService, AuditEvent
from dsl.governance.evidence_pack_service import EvidencePackService
from dsl.governance.governance_hooks_middleware import GovernanceContext, GovernanceHookType, GovernancePlane

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/consumption-contracts", tags=["API Consumption Contracts"])

# --- Enums and Data Models ---

class SLATier(Enum):
    T0_REGULATED = "T0"  # <100ms, 99.99% uptime, real-time alerts
    T1_ENTERPRISE = "T1"  # <500ms, 99.9% uptime, hourly alerts  
    T2_MIDMARKET = "T2"   # <2000ms, 99.5% uptime, daily alerts

class APIType(Enum):
    DASHBOARD = "dashboard"
    AGENT = "agent"
    PARTNER = "partner"
    REPORTING = "reporting"
    INTERNAL = "internal"

class RetryStrategy(Enum):
    EXPONENTIAL_BACKOFF = "exponential_backoff"
    LINEAR_BACKOFF = "linear_backoff"
    FIXED_INTERVAL = "fixed_interval"
    NO_RETRY = "no_retry"

class PersonaType(Enum):
    CRO = "cro"
    SALES_MANAGER = "sales_manager"
    REV_OPS = "rev_ops"
    COMPLIANCE = "compliance"
    DEVELOPER = "developer"

# --- Pydantic Models ---

class APIContractRequest(BaseModel):
    api_name: str = Field(..., description="Unique API name")
    api_type: APIType = Field(..., description="Type of API")
    sla_tier: SLATier = Field(SLATier.T2_MIDMARKET, description="SLA tier for the API")
    tenant_id: int = Field(..., description="Tenant ID for multi-tenant isolation")
    description: str = Field("", description="API description")
    version: str = Field("1.0.0", description="API version")
    endpoints: List[Dict[str, Any]] = Field([], description="List of API endpoints")
    rate_limits: Dict[str, int] = Field({}, description="Rate limiting configuration")
    retry_config: Dict[str, Any] = Field({}, description="Retry and backoff configuration")
    cost_attribution: Dict[str, Any] = Field({}, description="FinOps cost attribution settings")
    governance_metadata: Dict[str, Any] = Field({}, description="Governance and compliance metadata")

class APIRegistryEntry(BaseModel):
    api_id: str
    api_name: str
    api_type: APIType
    sla_tier: SLATier
    tenant_id: int
    version: str
    status: str
    endpoints: List[Dict[str, Any]]
    rate_limits: Dict[str, int]
    cost_metrics: Dict[str, Any]
    created_at: str
    updated_at: str

class RateLimitConfig(BaseModel):
    api_key: str = Field(..., description="API key for rate limiting")
    requests_per_minute: int = Field(100, description="Requests per minute limit")
    requests_per_hour: int = Field(1000, description="Requests per hour limit")
    requests_per_day: int = Field(10000, description="Requests per day limit")
    burst_limit: int = Field(50, description="Burst request limit")
    quota_reset_time: str = Field("00:00", description="Daily quota reset time (HH:MM)")

class RetryConfig(BaseModel):
    strategy: RetryStrategy = Field(RetryStrategy.EXPONENTIAL_BACKOFF, description="Retry strategy")
    max_attempts: int = Field(3, description="Maximum retry attempts")
    base_delay_ms: int = Field(1000, description="Base delay in milliseconds")
    max_delay_ms: int = Field(30000, description="Maximum delay in milliseconds")
    backoff_multiplier: float = Field(2.0, description="Backoff multiplier for exponential strategy")
    idempotency_enabled: bool = Field(True, description="Enable idempotency for safe retries")

class SLAMonitoringConfig(BaseModel):
    api_id: str = Field(..., description="API identifier")
    sla_tier: SLATier = Field(..., description="SLA tier")
    latency_threshold_ms: int = Field(..., description="Latency threshold in milliseconds")
    error_rate_threshold: float = Field(0.01, description="Error rate threshold (0.01 = 1%)")
    availability_threshold: float = Field(0.999, description="Availability threshold (0.999 = 99.9%)")
    alert_channels: List[str] = Field([], description="Alert notification channels")

class FinOpsConfig(BaseModel):
    api_id: str = Field(..., description="API identifier")
    cost_per_request: float = Field(0.001, description="Cost per API request")
    cost_per_gb_transfer: float = Field(0.10, description="Cost per GB data transfer")
    monthly_budget_limit: float = Field(1000.0, description="Monthly budget limit")
    budget_alert_thresholds: List[float] = Field([0.5, 0.8, 0.95], description="Budget alert thresholds")
    chargeback_enabled: bool = Field(True, description="Enable tenant chargeback")

class PersonaDashboardConfig(BaseModel):
    persona: PersonaType = Field(..., description="User persona type")
    dashboard_widgets: List[str] = Field(..., description="Dashboard widgets for persona")
    metrics_focus: List[str] = Field(..., description="Key metrics for persona")
    alert_preferences: Dict[str, Any] = Field({}, description="Alert preferences")

# --- In-memory stores (in production, these would be database-backed) ---

api_registry: Dict[str, APIRegistryEntry] = {}
rate_limit_configs: Dict[str, RateLimitConfig] = {}
retry_configs: Dict[str, RetryConfig] = {}
sla_monitoring_configs: Dict[str, SLAMonitoringConfig] = {}
finops_configs: Dict[str, FinOpsConfig] = {}
persona_dashboards: Dict[str, PersonaDashboardConfig] = {}

# Current usage tracking (in production, this would be Redis/database)
api_usage_tracking: Dict[str, Dict[str, Any]] = {}

# Dependency functions
def get_evidence_service() -> EvidencePackService:
    return EvidencePackService()

def get_audit_service() -> AuditTrailService:
    return AuditTrailService()

@router.post("/registry/register")
async def register_api_contract(
    request: APIContractRequest,
    evidence_service: EvidencePackService = Depends(get_evidence_service),
    audit_service: AuditTrailService = Depends(get_audit_service)
):
    """
    Task 11.4.3: Build API registry and discovery catalog
    Task 11.4.4: Define SLA tiers (T0/T1/T2) in API contracts
    """
    logger.info(f"Registering API contract: {request.api_name}")
    
    try:
        api_id = f"api_{request.api_name}_{uuid.uuid4().hex[:8]}"
        
        # Create registry entry
        registry_entry = APIRegistryEntry(
            api_id=api_id,
            api_name=request.api_name,
            api_type=request.api_type,
            sla_tier=request.sla_tier,
            tenant_id=request.tenant_id,
            version=request.version,
            status="active",
            endpoints=request.endpoints,
            rate_limits=request.rate_limits,
            cost_metrics={
                "requests_count": 0,
                "data_transfer_gb": 0.0,
                "total_cost": 0.0
            },
            created_at=datetime.now(timezone.utc).isoformat(),
            updated_at=datetime.now(timezone.utc).isoformat()
        )
        
        # Store in registry
        api_registry[api_id] = registry_entry
        
        # Configure SLA monitoring based on tier
        await configure_sla_monitoring(api_id, request.sla_tier)
        
        # Configure FinOps tracking
        await configure_finops_tracking(api_id, request.cost_attribution)
        
        # Create evidence pack
        await evidence_service.create_evidence_pack(
            tenant_id=request.tenant_id,
            event_type="API_CONTRACT_REGISTERED",
            event_data={
                "api_id": api_id,
                "api_contract": request.dict(),
                "registry_entry": registry_entry.dict()
            },
            governance_metadata=request.governance_metadata
        )
        
        # Log audit event
        await audit_service.log_event(
            AuditEvent(
                event_id=str(uuid.uuid4()),
                timestamp=datetime.now(timezone.utc).isoformat(),
                service_name="APIConsumptionContractsAPI",
                event_type="API_CONTRACT_REGISTERED",
                tenant_id=request.tenant_id,
                user_id="api_user",
                description=f"API contract registered: {request.api_name}",
                metadata={"api_id": api_id, "sla_tier": request.sla_tier.value}
            )
        )
        
        return {
            "success": True,
            "api_id": api_id,
            "message": f"API contract {request.api_name} registered successfully",
            "registry_entry": registry_entry.dict()
        }
        
    except Exception as e:
        logger.error(f"Failed to register API contract {request.api_name}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Registration failed: {str(e)}")

@router.get("/registry/discover")
async def discover_apis(
    api_type: Optional[APIType] = Query(None, description="Filter by API type"),
    sla_tier: Optional[SLATier] = Query(None, description="Filter by SLA tier"),
    tenant_id: Optional[int] = Query(None, description="Filter by tenant ID")
):
    """
    Task 11.4.3: Build API registry and discovery catalog
    Discover available APIs based on filters
    """
    try:
        filtered_apis = []
        
        for api_id, registry_entry in api_registry.items():
            # Apply filters
            if api_type and registry_entry.api_type != api_type:
                continue
            if sla_tier and registry_entry.sla_tier != sla_tier:
                continue
            if tenant_id and registry_entry.tenant_id != tenant_id:
                continue
            
            filtered_apis.append(registry_entry.dict())
        
        return {
            "total_apis": len(filtered_apis),
            "apis": filtered_apis,
            "filters_applied": {
                "api_type": api_type.value if api_type else None,
                "sla_tier": sla_tier.value if sla_tier else None,
                "tenant_id": tenant_id
            }
        }
        
    except Exception as e:
        logger.error(f"API discovery failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Discovery failed: {str(e)}")

@router.post("/rate-limiting/configure")
async def configure_rate_limiting(
    config: RateLimitConfig,
    audit_service: AuditTrailService = Depends(get_audit_service)
):
    """
    Task 11.4.18: Add rate limiting and quotas per API key
    Configure rate limiting for API keys
    """
    logger.info(f"Configuring rate limiting for API key: {config.api_key}")
    
    try:
        # Store rate limit configuration
        rate_limit_configs[config.api_key] = config
        
        # Initialize usage tracking for this API key
        api_usage_tracking[config.api_key] = {
            "requests_this_minute": 0,
            "requests_this_hour": 0,
            "requests_this_day": 0,
            "last_request_time": None,
            "quota_reset_time": config.quota_reset_time
        }
        
        # Log configuration
        await audit_service.log_event(
            AuditEvent(
                event_id=str(uuid.uuid4()),
                timestamp=datetime.now(timezone.utc).isoformat(),
                service_name="APIConsumptionContractsAPI",
                event_type="RATE_LIMITING_CONFIGURED",
                tenant_id=0,  # System-level configuration
                user_id="api_admin",
                description=f"Rate limiting configured for API key {config.api_key}",
                metadata={"rate_limit_config": config.dict()}
            )
        )
        
        return {
            "success": True,
            "api_key": config.api_key,
            "message": "Rate limiting configured successfully",
            "configuration": config.dict()
        }
        
    except Exception as e:
        logger.error(f"Rate limiting configuration failed for {config.api_key}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Configuration failed: {str(e)}")

@router.post("/retry/configure")
async def configure_retry_policy(
    api_id: str,
    config: RetryConfig,
    audit_service: AuditTrailService = Depends(get_audit_service)
):
    """
    Task 11.4.17: Configure retry/backoff + idempotency for APIs
    Configure retry policies for APIs
    """
    logger.info(f"Configuring retry policy for API: {api_id}")
    
    try:
        # Validate API exists
        if api_id not in api_registry:
            raise HTTPException(status_code=404, detail=f"API {api_id} not found in registry")
        
        # Store retry configuration
        retry_configs[api_id] = config
        
        # Log configuration
        await audit_service.log_event(
            AuditEvent(
                event_id=str(uuid.uuid4()),
                timestamp=datetime.now(timezone.utc).isoformat(),
                service_name="APIConsumptionContractsAPI",
                event_type="RETRY_POLICY_CONFIGURED",
                tenant_id=api_registry[api_id].tenant_id,
                user_id="api_admin",
                description=f"Retry policy configured for API {api_id}",
                metadata={"api_id": api_id, "retry_config": config.dict()}
            )
        )
        
        return {
            "success": True,
            "api_id": api_id,
            "message": "Retry policy configured successfully",
            "configuration": config.dict()
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Retry policy configuration failed for {api_id}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Configuration failed: {str(e)}")

@router.post("/finops/configure")
async def configure_finops_tracking(
    api_id: str,
    cost_attribution: Dict[str, Any],
    audit_service: AuditTrailService = Depends(get_audit_service)
):
    """
    Task 11.4.20: Configure FinOps cost attribution per API
    Task 11.4.21: Create budget alerts for API usage
    """
    logger.info(f"Configuring FinOps tracking for API: {api_id}")
    
    try:
        # Validate API exists
        if api_id not in api_registry:
            raise HTTPException(status_code=404, detail=f"API {api_id} not found in registry")
        
        # Create FinOps configuration
        finops_config = FinOpsConfig(
            api_id=api_id,
            cost_per_request=cost_attribution.get("cost_per_request", 0.001),
            cost_per_gb_transfer=cost_attribution.get("cost_per_gb_transfer", 0.10),
            monthly_budget_limit=cost_attribution.get("monthly_budget_limit", 1000.0),
            budget_alert_thresholds=cost_attribution.get("budget_alert_thresholds", [0.5, 0.8, 0.95]),
            chargeback_enabled=cost_attribution.get("chargeback_enabled", True)
        )
        
        # Store configuration
        finops_configs[api_id] = finops_config
        
        # Log configuration
        await audit_service.log_event(
            AuditEvent(
                event_id=str(uuid.uuid4()),
                timestamp=datetime.now(timezone.utc).isoformat(),
                service_name="APIConsumptionContractsAPI",
                event_type="FINOPS_TRACKING_CONFIGURED",
                tenant_id=api_registry[api_id].tenant_id,
                user_id="api_admin",
                description=f"FinOps tracking configured for API {api_id}",
                metadata={"api_id": api_id, "finops_config": finops_config.dict()}
            )
        )
        
        return {
            "success": True,
            "api_id": api_id,
            "message": "FinOps tracking configured successfully",
            "configuration": finops_config.dict()
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"FinOps configuration failed for {api_id}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Configuration failed: {str(e)}")

@router.post("/sla-monitoring/configure")
async def configure_sla_monitoring(
    api_id: str,
    sla_tier: SLATier,
    audit_service: AuditTrailService = Depends(get_audit_service)
):
    """
    Task 11.4.27: Configure SLA-aware monitoring on APIs
    Configure SLA monitoring based on tier
    """
    logger.info(f"Configuring SLA monitoring for API: {api_id} with tier {sla_tier.value}")
    
    try:
        # Define SLA requirements by tier
        sla_requirements = {
            SLATier.T0_REGULATED: {
                "latency_threshold_ms": 100,
                "error_rate_threshold": 0.001,  # 0.1%
                "availability_threshold": 0.9999,  # 99.99%
                "alert_channels": ["email", "slack", "pagerduty"]
            },
            SLATier.T1_ENTERPRISE: {
                "latency_threshold_ms": 500,
                "error_rate_threshold": 0.01,  # 1%
                "availability_threshold": 0.999,  # 99.9%
                "alert_channels": ["email", "slack"]
            },
            SLATier.T2_MIDMARKET: {
                "latency_threshold_ms": 2000,
                "error_rate_threshold": 0.05,  # 5%
                "availability_threshold": 0.995,  # 99.5%
                "alert_channels": ["email"]
            }
        }
        
        requirements = sla_requirements[sla_tier]
        
        # Create SLA monitoring configuration
        sla_config = SLAMonitoringConfig(
            api_id=api_id,
            sla_tier=sla_tier,
            **requirements
        )
        
        # Store configuration
        sla_monitoring_configs[api_id] = sla_config
        
        # Log configuration
        if audit_service:
            await audit_service.log_event(
                AuditEvent(
                    event_id=str(uuid.uuid4()),
                    timestamp=datetime.now(timezone.utc).isoformat(),
                    service_name="APIConsumptionContractsAPI",
                    event_type="SLA_MONITORING_CONFIGURED",
                    tenant_id=api_registry.get(api_id, APIRegistryEntry(tenant_id=0, api_id="", api_name="", api_type=APIType.INTERNAL, sla_tier=SLATier.T2_MIDMARKET, version="", status="", endpoints=[], rate_limits={}, cost_metrics={}, created_at="", updated_at="")).tenant_id,
                    user_id="api_admin",
                    description=f"SLA monitoring configured for API {api_id}",
                    metadata={"api_id": api_id, "sla_config": sla_config.dict()}
                )
            )
        
        return {
            "success": True,
            "api_id": api_id,
            "sla_tier": sla_tier.value,
            "message": "SLA monitoring configured successfully",
            "configuration": sla_config.dict()
        }
        
    except Exception as e:
        logger.error(f"SLA monitoring configuration failed for {api_id}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Configuration failed: {str(e)}")

@router.post("/persona-dashboard/configure")
async def configure_persona_dashboard(
    config: PersonaDashboardConfig,
    audit_service: AuditTrailService = Depends(get_audit_service)
):
    """
    Task 11.4.28: Build persona dashboards for API usage
    Configure persona-specific dashboards
    """
    logger.info(f"Configuring persona dashboard for: {config.persona.value}")
    
    try:
        # Define default configurations for each persona
        persona_defaults = {
            PersonaType.CRO: {
                "dashboard_widgets": ["api_usage_summary", "cost_overview", "sla_compliance", "revenue_impact"],
                "metrics_focus": ["total_api_calls", "cost_per_tenant", "sla_violations", "business_impact"],
                "alert_preferences": {"frequency": "real_time", "channels": ["email", "slack"]}
            },
            PersonaType.SALES_MANAGER: {
                "dashboard_widgets": ["team_api_usage", "quota_tracking", "performance_metrics"],
                "metrics_focus": ["team_usage", "quota_consumption", "performance_trends"],
                "alert_preferences": {"frequency": "daily", "channels": ["email"]}
            },
            PersonaType.REV_OPS: {
                "dashboard_widgets": ["operational_metrics", "system_health", "data_quality", "pipeline_status"],
                "metrics_focus": ["system_uptime", "data_freshness", "pipeline_success_rate"],
                "alert_preferences": {"frequency": "real_time", "channels": ["slack", "pagerduty"]}
            },
            PersonaType.COMPLIANCE: {
                "dashboard_widgets": ["audit_logs", "compliance_status", "data_governance", "risk_metrics"],
                "metrics_focus": ["audit_trail_completeness", "policy_violations", "data_residency_compliance"],
                "alert_preferences": {"frequency": "real_time", "channels": ["email", "compliance_portal"]}
            },
            PersonaType.DEVELOPER: {
                "dashboard_widgets": ["api_performance", "error_rates", "usage_patterns", "documentation"],
                "metrics_focus": ["response_times", "error_rates", "usage_analytics"],
                "alert_preferences": {"frequency": "real_time", "channels": ["slack", "webhook"]}
            }
        }
        
        # Merge with defaults
        defaults = persona_defaults.get(config.persona, {})
        if not config.dashboard_widgets:
            config.dashboard_widgets = defaults.get("dashboard_widgets", [])
        if not config.metrics_focus:
            config.metrics_focus = defaults.get("metrics_focus", [])
        if not config.alert_preferences:
            config.alert_preferences = defaults.get("alert_preferences", {})
        
        # Store configuration
        persona_dashboards[config.persona.value] = config
        
        # Log configuration
        await audit_service.log_event(
            AuditEvent(
                event_id=str(uuid.uuid4()),
                timestamp=datetime.now(timezone.utc).isoformat(),
                service_name="APIConsumptionContractsAPI",
                event_type="PERSONA_DASHBOARD_CONFIGURED",
                tenant_id=0,  # System-level configuration
                user_id="api_admin",
                description=f"Persona dashboard configured for {config.persona.value}",
                metadata={"persona": config.persona.value, "dashboard_config": config.dict()}
            )
        )
        
        return {
            "success": True,
            "persona": config.persona.value,
            "message": "Persona dashboard configured successfully",
            "configuration": config.dict()
        }
        
    except Exception as e:
        logger.error(f"Persona dashboard configuration failed for {config.persona.value}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Configuration failed: {str(e)}")

@router.get("/documentation/generate/{api_id}")
async def generate_api_documentation(api_id: str):
    """
    Task 11.4.23: Create API documentation generator
    Generate comprehensive API documentation
    """
    logger.info(f"Generating documentation for API: {api_id}")
    
    try:
        # Validate API exists
        if api_id not in api_registry:
            raise HTTPException(status_code=404, detail=f"API {api_id} not found in registry")
        
        registry_entry = api_registry[api_id]
        
        # Generate comprehensive documentation
        documentation = {
            "api_id": api_id,
            "api_name": registry_entry.api_name,
            "version": registry_entry.version,
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "overview": {
                "description": f"API documentation for {registry_entry.api_name}",
                "api_type": registry_entry.api_type.value,
                "sla_tier": registry_entry.sla_tier.value,
                "status": registry_entry.status
            },
            "endpoints": registry_entry.endpoints,
            "authentication": {
                "type": "Bearer Token",
                "required_headers": ["Authorization", "X-Tenant-ID"],
                "rate_limiting": rate_limit_configs.get(api_id, {})
            },
            "sla_requirements": sla_monitoring_configs.get(api_id, {}).dict() if api_id in sla_monitoring_configs else {},
            "retry_policy": retry_configs.get(api_id, {}).dict() if api_id in retry_configs else {},
            "cost_information": finops_configs.get(api_id, {}).dict() if api_id in finops_configs else {},
            "governance": {
                "tenant_isolation": True,
                "audit_logging": True,
                "evidence_packs": True,
                "data_residency": "Configurable per tenant"
            },
            "examples": {
                "curl_example": f"curl -H 'Authorization: Bearer <token>' -H 'X-Tenant-ID: <tenant>' https://api.revai.pro/{api_id}/",
                "response_format": "JSON",
                "error_codes": ["400", "401", "403", "404", "429", "500"]
            }
        }
        
        return {
            "success": True,
            "api_id": api_id,
            "documentation": documentation,
            "download_formats": ["json", "yaml", "markdown", "pdf"]
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Documentation generation failed for {api_id}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Documentation generation failed: {str(e)}")

@router.get("/monitoring/dashboard/{persona}")
async def get_persona_dashboard(persona: PersonaType):
    """
    Task 11.4.28: Build persona dashboards for API usage
    Get persona-specific dashboard data
    """
    try:
        # Get persona configuration
        persona_config = persona_dashboards.get(persona.value)
        if not persona_config:
            # Return default configuration
            await configure_persona_dashboard(PersonaDashboardConfig(persona=persona, dashboard_widgets=[], metrics_focus=[]))
            persona_config = persona_dashboards.get(persona.value)
        
        # Generate mock dashboard data based on persona
        dashboard_data = {
            "persona": persona.value,
            "dashboard_config": persona_config.dict() if persona_config else {},
            "widgets": {},
            "last_updated": datetime.now(timezone.utc).isoformat()
        }
        
        # Add persona-specific widget data
        if persona == PersonaType.CRO:
            dashboard_data["widgets"] = {
                "api_usage_summary": {"total_calls": 1500000, "growth_rate": 0.15},
                "cost_overview": {"monthly_cost": 25000, "cost_per_call": 0.0167},
                "sla_compliance": {"uptime": 99.95, "violations": 2},
                "revenue_impact": {"api_driven_revenue": 500000}
            }
        elif persona == PersonaType.COMPLIANCE:
            dashboard_data["widgets"] = {
                "audit_logs": {"events_logged": 50000, "compliance_score": 98.5},
                "compliance_status": {"policies_enforced": 25, "violations": 1},
                "data_governance": {"pii_fields_masked": 100, "consent_compliance": 99.8},
                "risk_metrics": {"risk_score": 2.1, "critical_issues": 0}
            }
        elif persona == PersonaType.DEVELOPER:
            dashboard_data["widgets"] = {
                "api_performance": {"avg_response_time": 245, "p99_latency": 850},
                "error_rates": {"error_rate": 0.02, "5xx_errors": 15},
                "usage_patterns": {"peak_hour": "14:00", "usage_trend": "increasing"},
                "documentation": {"docs_updated": "2024-12-19", "coverage": 95}
            }
        
        return dashboard_data
        
    except Exception as e:
        logger.error(f"Dashboard retrieval failed for persona {persona.value}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Dashboard retrieval failed: {str(e)}")

@router.get("/health")
async def health_check():
    """Health check endpoint for API consumption contracts service"""
    return {
        "status": "healthy",
        "service": "APIConsumptionContractsAPI",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "version": "1.0.0",
        "registry_size": len(api_registry),
        "active_configs": {
            "rate_limits": len(rate_limit_configs),
            "retry_policies": len(retry_configs),
            "sla_monitoring": len(sla_monitoring_configs),
            "finops_tracking": len(finops_configs),
            "persona_dashboards": len(persona_dashboards)
        }
    }
