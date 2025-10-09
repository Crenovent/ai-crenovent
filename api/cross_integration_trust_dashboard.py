"""
Task 4.5.27: Create cross-integration trust dashboards
- Unified view of success rates, errors, overrides across all integrations
- Real-time health monitoring
- Ops visibility with tenant isolation enforced
"""

from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel, Field
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
from enum import Enum
import uuid
import logging

app = FastAPI(title="RBIA Cross-Integration Trust Dashboard")
logger = logging.getLogger(__name__)

class IntegrationType(str, Enum):
    CRM = "crm"
    BILLING = "billing"
    CONTRACT = "contract"
    MARKETING = "marketing"
    FORECAST = "forecast"
    COMPENSATION = "compensation"
    DASHBOARD = "dashboard"
    KNOWLEDGE_GRAPH = "knowledge_graph"

class IntegrationStatus(str, Enum):
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    DOWN = "down"
    MAINTENANCE = "maintenance"

class IntegrationHealth(BaseModel):
    integration_type: IntegrationType
    integration_name: str
    status: IntegrationStatus
    
    # Success metrics
    total_calls: int = 0
    successful_calls: int = 0
    failed_calls: int = 0
    success_rate: float = 0.0
    
    # Performance metrics
    avg_latency_ms: float = 0.0
    p95_latency_ms: float = 0.0
    p99_latency_ms: float = 0.0
    
    # Error tracking
    error_count: int = 0
    error_rate: float = 0.0
    recent_errors: List[str] = Field(default_factory=list)
    
    # Override tracking
    override_count: int = 0
    override_rate: float = 0.0
    
    # Fallback tracking
    fallback_triggered: int = 0
    fallback_success_rate: float = 0.0
    
    # Trust score
    trust_score: float = 0.0
    confidence_score: float = 0.0
    
    # Last activity
    last_success: Optional[datetime] = None
    last_failure: Optional[datetime] = None
    last_checked: datetime = Field(default_factory=datetime.utcnow)

class CrossIntegrationDashboard(BaseModel):
    dashboard_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    tenant_id: str
    
    # Overall health
    overall_status: IntegrationStatus
    overall_success_rate: float = 0.0
    overall_trust_score: float = 0.0
    
    # Integration breakdown
    integrations: List[IntegrationHealth] = Field(default_factory=list)
    
    # Aggregate metrics
    total_calls_24h: int = 0
    successful_calls_24h: int = 0
    failed_calls_24h: int = 0
    
    # Error summary
    top_errors: List[Dict[str, Any]] = Field(default_factory=list)
    error_hotspots: List[str] = Field(default_factory=list)
    
    # Override summary
    total_overrides: int = 0
    override_by_type: Dict[str, int] = Field(default_factory=dict)
    
    # Performance summary
    slowest_integrations: List[Dict[str, Any]] = Field(default_factory=list)
    fastest_integrations: List[Dict[str, Any]] = Field(default_factory=list)
    
    # Time window
    time_window_hours: int = 24
    generated_at: datetime = Field(default_factory=datetime.utcnow)

class IntegrationAlert(BaseModel):
    alert_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    tenant_id: str
    integration_type: IntegrationType
    integration_name: str
    
    severity: str  # critical, high, medium, low
    alert_type: str  # error_spike, latency_spike, success_rate_drop, override_spike
    
    message: str
    details: Dict[str, Any] = Field(default_factory=dict)
    
    triggered_at: datetime = Field(default_factory=datetime.utcnow)
    acknowledged: bool = False
    resolved: bool = False

class TrustTrend(BaseModel):
    trend_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    tenant_id: str
    
    # Trend data points (time series)
    timestamps: List[datetime] = Field(default_factory=list)
    trust_scores: List[float] = Field(default_factory=list)
    success_rates: List[float] = Field(default_factory=list)
    error_rates: List[float] = Field(default_factory=list)
    
    # Trend analysis
    trend_direction: str  # improving, stable, degrading
    trend_percentage: float = 0.0
    
    generated_at: datetime = Field(default_factory=datetime.utcnow)

# In-memory storage
dashboards_store: Dict[str, CrossIntegrationDashboard] = {}
integration_health_store: Dict[str, List[IntegrationHealth]] = {}
alerts_store: Dict[str, List[IntegrationAlert]] = {}
trends_store: Dict[str, TrustTrend] = {}

class CrossIntegrationTrustService:
    """Service for managing cross-integration trust metrics"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    def calculate_integration_health(self, tenant_id: str) -> List[IntegrationHealth]:
        """Calculate health metrics for all integrations"""
        
        # Mock data - in production, query from traces and metrics DB
        integrations = [
            IntegrationHealth(
                integration_type=IntegrationType.CRM,
                integration_name="Salesforce",
                status=IntegrationStatus.HEALTHY,
                total_calls=1000,
                successful_calls=980,
                failed_calls=20,
                success_rate=98.0,
                avg_latency_ms=250.5,
                p95_latency_ms=450.0,
                p99_latency_ms=650.0,
                error_count=20,
                error_rate=2.0,
                override_count=5,
                override_rate=0.5,
                fallback_triggered=3,
                fallback_success_rate=100.0,
                trust_score=0.96,
                confidence_score=0.94,
                last_success=datetime.utcnow() - timedelta(minutes=5),
                last_failure=datetime.utcnow() - timedelta(hours=2)
            ),
            IntegrationHealth(
                integration_type=IntegrationType.CRM,
                integration_name="HubSpot",
                status=IntegrationStatus.HEALTHY,
                total_calls=500,
                successful_calls=495,
                failed_calls=5,
                success_rate=99.0,
                avg_latency_ms=180.2,
                p95_latency_ms=320.0,
                p99_latency_ms=450.0,
                error_count=5,
                error_rate=1.0,
                override_count=2,
                override_rate=0.4,
                fallback_triggered=1,
                fallback_success_rate=100.0,
                trust_score=0.98,
                confidence_score=0.96,
                last_success=datetime.utcnow() - timedelta(minutes=2),
                last_failure=datetime.utcnow() - timedelta(hours=5)
            ),
            IntegrationHealth(
                integration_type=IntegrationType.BILLING,
                integration_name="Stripe",
                status=IntegrationStatus.DEGRADED,
                total_calls=300,
                successful_calls=270,
                failed_calls=30,
                success_rate=90.0,
                avg_latency_ms=450.8,
                p95_latency_ms=800.0,
                p99_latency_ms=1200.0,
                error_count=30,
                error_rate=10.0,
                override_count=8,
                override_rate=2.7,
                fallback_triggered=10,
                fallback_success_rate=80.0,
                trust_score=0.85,
                confidence_score=0.82,
                last_success=datetime.utcnow() - timedelta(minutes=10),
                last_failure=datetime.utcnow() - timedelta(minutes=3)
            ),
            IntegrationHealth(
                integration_type=IntegrationType.FORECAST,
                integration_name="Internal Forecast Module",
                status=IntegrationStatus.HEALTHY,
                total_calls=2000,
                successful_calls=1990,
                failed_calls=10,
                success_rate=99.5,
                avg_latency_ms=120.3,
                p95_latency_ms=200.0,
                p99_latency_ms=300.0,
                error_count=10,
                error_rate=0.5,
                override_count=15,
                override_rate=0.75,
                fallback_triggered=5,
                fallback_success_rate=100.0,
                trust_score=0.99,
                confidence_score=0.97,
                last_success=datetime.utcnow() - timedelta(seconds=30),
                last_failure=datetime.utcnow() - timedelta(hours=6)
            ),
            IntegrationHealth(
                integration_type=IntegrationType.KNOWLEDGE_GRAPH,
                integration_name="Knowledge Graph Store",
                status=IntegrationStatus.HEALTHY,
                total_calls=5000,
                successful_calls=4975,
                failed_calls=25,
                success_rate=99.5,
                avg_latency_ms=95.2,
                p95_latency_ms=150.0,
                p99_latency_ms=220.0,
                error_count=25,
                error_rate=0.5,
                override_count=0,
                override_rate=0.0,
                fallback_triggered=0,
                fallback_success_rate=0.0,
                trust_score=0.99,
                confidence_score=0.98,
                last_success=datetime.utcnow() - timedelta(seconds=10),
                last_failure=datetime.utcnow() - timedelta(hours=8)
            )
        ]
        
        return integrations
    
    def build_dashboard(self, tenant_id: str, time_window_hours: int = 24) -> CrossIntegrationDashboard:
        """Build comprehensive cross-integration dashboard"""
        
        integrations = self.calculate_integration_health(tenant_id)
        
        # Calculate overall metrics
        total_calls = sum(i.total_calls for i in integrations)
        successful_calls = sum(i.successful_calls for i in integrations)
        failed_calls = sum(i.failed_calls for i in integrations)
        
        overall_success_rate = (successful_calls / total_calls * 100) if total_calls > 0 else 0.0
        overall_trust_score = sum(i.trust_score for i in integrations) / len(integrations) if integrations else 0.0
        
        # Determine overall status
        degraded_count = sum(1 for i in integrations if i.status == IntegrationStatus.DEGRADED)
        down_count = sum(1 for i in integrations if i.status == IntegrationStatus.DOWN)
        
        if down_count > 0:
            overall_status = IntegrationStatus.DOWN
        elif degraded_count > 0:
            overall_status = IntegrationStatus.DEGRADED
        else:
            overall_status = IntegrationStatus.HEALTHY
        
        # Top errors
        top_errors = [
            {
                "error": "Connection timeout",
                "count": 15,
                "integration": "Stripe"
            },
            {
                "error": "Authentication failed",
                "count": 10,
                "integration": "Salesforce"
            },
            {
                "error": "Rate limit exceeded",
                "count": 5,
                "integration": "HubSpot"
            }
        ]
        
        # Error hotspots
        error_hotspots = ["Stripe", "Salesforce"]
        
        # Override summary
        total_overrides = sum(i.override_count for i in integrations)
        override_by_type = {
            "CRM": 7,
            "BILLING": 8,
            "FORECAST": 15
        }
        
        # Performance rankings
        sorted_by_latency = sorted(integrations, key=lambda x: x.avg_latency_ms)
        slowest = [{"name": i.integration_name, "latency_ms": i.avg_latency_ms} for i in sorted_by_latency[-3:]]
        fastest = [{"name": i.integration_name, "latency_ms": i.avg_latency_ms} for i in sorted_by_latency[:3]]
        
        dashboard = CrossIntegrationDashboard(
            tenant_id=tenant_id,
            overall_status=overall_status,
            overall_success_rate=overall_success_rate,
            overall_trust_score=overall_trust_score,
            integrations=integrations,
            total_calls_24h=total_calls,
            successful_calls_24h=successful_calls,
            failed_calls_24h=failed_calls,
            top_errors=top_errors,
            error_hotspots=error_hotspots,
            total_overrides=total_overrides,
            override_by_type=override_by_type,
            slowest_integrations=slowest,
            fastest_integrations=fastest,
            time_window_hours=time_window_hours
        )
        
        return dashboard

trust_service = CrossIntegrationTrustService()

@app.get("/cross-integration/dashboard", response_model=CrossIntegrationDashboard)
async def get_cross_integration_dashboard(
    tenant_id: str,
    time_window_hours: int = Query(24, ge=1, le=168)
):
    """
    Get unified cross-integration trust dashboard
    Shows success rates, errors, overrides across all integrations
    """
    try:
        dashboard = trust_service.build_dashboard(tenant_id, time_window_hours)
        dashboards_store[dashboard.dashboard_id] = dashboard
        
        logger.info(f"Generated cross-integration dashboard for tenant {tenant_id}")
        return dashboard
        
    except Exception as e:
        logger.error(f"Failed to generate dashboard: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/cross-integration/health/{integration_type}", response_model=List[IntegrationHealth])
async def get_integration_health(
    tenant_id: str,
    integration_type: IntegrationType
):
    """Get health metrics for specific integration type"""
    try:
        all_integrations = trust_service.calculate_integration_health(tenant_id)
        filtered = [i for i in all_integrations if i.integration_type == integration_type]
        
        return filtered
        
    except Exception as e:
        logger.error(f"Failed to get integration health: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/cross-integration/alerts", response_model=List[IntegrationAlert])
async def get_integration_alerts(
    tenant_id: str,
    severity: Optional[str] = None,
    unresolved_only: bool = True
):
    """Get integration alerts for tenant"""
    
    # Mock alerts
    alerts = [
        IntegrationAlert(
            tenant_id=tenant_id,
            integration_type=IntegrationType.BILLING,
            integration_name="Stripe",
            severity="high",
            alert_type="error_spike",
            message="Error rate increased to 10% in last hour",
            details={"error_rate": 10.0, "threshold": 5.0},
            acknowledged=False,
            resolved=False
        ),
        IntegrationAlert(
            tenant_id=tenant_id,
            integration_type=IntegrationType.CRM,
            integration_name="Salesforce",
            severity="medium",
            alert_type="latency_spike",
            message="P95 latency increased to 450ms",
            details={"p95_latency_ms": 450, "baseline": 300},
            acknowledged=True,
            resolved=False
        )
    ]
    
    # Filter by severity
    if severity:
        alerts = [a for a in alerts if a.severity == severity]
    
    # Filter unresolved
    if unresolved_only:
        alerts = [a for a in alerts if not a.resolved]
    
    return alerts

@app.get("/cross-integration/trends", response_model=TrustTrend)
async def get_trust_trends(
    tenant_id: str,
    days: int = Query(7, ge=1, le=30)
):
    """Get trust score trends over time"""
    
    # Generate mock time series data
    now = datetime.utcnow()
    timestamps = [now - timedelta(hours=i*6) for i in range(days*4)]
    timestamps.reverse()
    
    # Mock trend data (improving trend)
    base_trust = 0.90
    trust_scores = [base_trust + (i * 0.002) for i in range(len(timestamps))]
    
    base_success = 95.0
    success_rates = [base_success + (i * 0.1) for i in range(len(timestamps))]
    
    base_error = 5.0
    error_rates = [base_error - (i * 0.05) for i in range(len(timestamps))]
    
    trend = TrustTrend(
        tenant_id=tenant_id,
        timestamps=timestamps,
        trust_scores=trust_scores,
        success_rates=success_rates,
        error_rates=error_rates,
        trend_direction="improving",
        trend_percentage=2.5
    )
    
    return trend

@app.post("/cross-integration/alerts/{alert_id}/acknowledge")
async def acknowledge_alert(alert_id: str, tenant_id: str):
    """Acknowledge an integration alert"""
    
    return {
        "status": "acknowledged",
        "alert_id": alert_id,
        "acknowledged_at": datetime.utcnow().isoformat()
    }

@app.post("/cross-integration/alerts/{alert_id}/resolve")
async def resolve_alert(alert_id: str, tenant_id: str, resolution_notes: str):
    """Resolve an integration alert"""
    
    return {
        "status": "resolved",
        "alert_id": alert_id,
        "resolved_at": datetime.utcnow().isoformat(),
        "resolution_notes": resolution_notes
    }

@app.get("/cross-integration/summary")
async def get_integration_summary(tenant_id: str):
    """Get high-level integration summary"""
    
    dashboard = trust_service.build_dashboard(tenant_id)
    
    return {
        "tenant_id": tenant_id,
        "overall_status": dashboard.overall_status,
        "overall_success_rate": dashboard.overall_success_rate,
        "overall_trust_score": dashboard.overall_trust_score,
        "total_integrations": len(dashboard.integrations),
        "healthy_integrations": sum(1 for i in dashboard.integrations if i.status == IntegrationStatus.HEALTHY),
        "degraded_integrations": sum(1 for i in dashboard.integrations if i.status == IntegrationStatus.DEGRADED),
        "down_integrations": sum(1 for i in dashboard.integrations if i.status == IntegrationStatus.DOWN),
        "total_calls_24h": dashboard.total_calls_24h,
        "error_hotspots": dashboard.error_hotspots,
        "generated_at": datetime.utcnow().isoformat()
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8027)

