"""
Task 4.5.41: Provide multi-tenant incident transparency portal
- Public-facing incident status page
- Per-tenant incident visibility
- Integration outage notifications
- Historical incident dashboard
- SOC-ready multi-tenant view
- Real-time status updates
"""

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from pydantic import BaseModel, Field
from typing import Dict, List, Any, Optional, Set
from datetime import datetime, timedelta
from enum import Enum
import uuid
import logging
import asyncio

app = FastAPI(title="RBIA Incident Transparency Portal")
logger = logging.getLogger(__name__)

class ServiceComponent(str, Enum):
    CRM_INTEGRATION = "crm_integration"
    BILLING_INTEGRATION = "billing_integration"
    CONTRACT_INTEGRATION = "contract_integration"
    FORECAST_MODULE = "forecast_module"
    COMPENSATION_MODULE = "compensation_module"
    KNOWLEDGE_GRAPH = "knowledge_graph"
    API_GATEWAY = "api_gateway"
    AUTH_SERVICE = "auth_service"
    DATABASE = "database"
    CACHE_LAYER = "cache_layer"

class ComponentStatus(str, Enum):
    OPERATIONAL = "operational"
    DEGRADED_PERFORMANCE = "degraded_performance"
    PARTIAL_OUTAGE = "partial_outage"
    MAJOR_OUTAGE = "major_outage"
    MAINTENANCE = "maintenance"

class IncidentSeverity(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

class IncidentStatus(str, Enum):
    INVESTIGATING = "investigating"
    IDENTIFIED = "identified"
    MONITORING = "monitoring"
    RESOLVED = "resolved"

class ImpactScope(str, Enum):
    SINGLE_TENANT = "single_tenant"
    MULTIPLE_TENANTS = "multiple_tenants"
    ALL_TENANTS = "all_tenants"
    REGION_SPECIFIC = "region_specific"

class ServiceComponentHealth(BaseModel):
    component: ServiceComponent
    status: ComponentStatus
    
    # Metrics
    uptime_percentage: float = 100.0
    response_time_ms: float = 0.0
    error_rate: float = 0.0
    
    # Last incident
    last_incident: Optional[str] = None
    last_incident_at: Optional[datetime] = None
    
    # Metadata
    affected_tenants: int = 0
    maintenance_window: Optional[Dict[str, Any]] = None
    
    updated_at: datetime = Field(default_factory=datetime.utcnow)

class Incident(BaseModel):
    incident_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    
    # Basic information
    title: str
    description: str
    severity: IncidentSeverity
    status: IncidentStatus
    
    # Affected resources
    affected_components: List[ServiceComponent] = Field(default_factory=list)
    impact_scope: ImpactScope
    affected_tenant_ids: List[str] = Field(default_factory=list)
    affected_tenant_count: int = 0
    
    # Regional impact
    affected_regions: List[str] = Field(default_factory=list)
    
    # Timeline
    detected_at: datetime = Field(default_factory=datetime.utcnow)
    acknowledged_at: Optional[datetime] = None
    resolved_at: Optional[datetime] = None
    
    # Updates
    updates: List[Dict[str, Any]] = Field(default_factory=list)
    
    # Root cause (after resolution)
    root_cause: Optional[str] = None
    remediation_steps: List[str] = Field(default_factory=list)
    
    # SLA tracking
    sla_breached: bool = False
    time_to_acknowledge_minutes: Optional[float] = None
    time_to_resolve_minutes: Optional[float] = None
    
    # Notifications
    notifications_sent: int = 0
    subscribers_notified: List[str] = Field(default_factory=list)

class TenantIncidentView(BaseModel):
    tenant_id: str
    tenant_name: str
    
    # Current status
    overall_status: ComponentStatus
    active_incidents: List[Incident] = Field(default_factory=list)
    
    # Component health (tenant-specific)
    component_health: List[ServiceComponentHealth] = Field(default_factory=list)
    
    # Historical data
    incidents_last_30_days: int = 0
    uptime_last_30_days: float = 99.9
    mttr_minutes: float = 0.0  # Mean Time To Resolution
    
    # SLA status
    sla_status: str = "compliant"
    sla_breaches_this_month: int = 0
    
    last_updated: datetime = Field(default_factory=datetime.utcnow)

class SystemStatusPage(BaseModel):
    page_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    
    # Overall status
    overall_system_status: ComponentStatus
    active_incidents_count: int = 0
    
    # Component statuses
    components: List[ServiceComponentHealth] = Field(default_factory=list)
    
    # Active incidents
    active_incidents: List[Incident] = Field(default_factory=list)
    
    # Historical metrics
    uptime_7_days: float = 99.9
    uptime_30_days: float = 99.9
    uptime_90_days: float = 99.9
    
    # Recent incidents
    recent_incidents: List[Incident] = Field(default_factory=list)
    
    # Scheduled maintenance
    scheduled_maintenance: List[Dict[str, Any]] = Field(default_factory=list)
    
    generated_at: datetime = Field(default_factory=datetime.utcnow)

class IncidentNotification(BaseModel):
    notification_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    incident_id: str
    tenant_id: str
    
    notification_type: str  # email, sms, webhook, in_app
    recipient: str
    
    subject: str
    message: str
    
    sent_at: datetime = Field(default_factory=datetime.utcnow)
    acknowledged: bool = False
    acknowledged_at: Optional[datetime] = None

class IncidentSubscription(BaseModel):
    subscription_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    tenant_id: str
    user_email: str
    
    # Subscription preferences
    notify_on_incidents: bool = True
    notify_on_maintenance: bool = True
    notify_on_resolution: bool = True
    
    # Filter by severity
    min_severity: IncidentSeverity = IncidentSeverity.MEDIUM
    
    # Notification channels
    channels: List[str] = Field(default=["email"])
    
    created_at: datetime = Field(default_factory=datetime.utcnow)
    active: bool = True

# In-memory storage
components_store: Dict[ServiceComponent, ServiceComponentHealth] = {}
incidents_store: Dict[str, Incident] = {}
tenant_views_store: Dict[str, TenantIncidentView] = {}
subscriptions_store: Dict[str, IncidentSubscription] = {}
notifications_store: Dict[str, IncidentNotification] = {}

# WebSocket connections for real-time updates
active_connections: Set[WebSocket] = set()

class IncidentTransparencyService:
    """Service for managing incident transparency"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self._init_default_components()
    
    def _init_default_components(self):
        """Initialize default component health status"""
        
        for component in ServiceComponent:
            components_store[component] = ServiceComponentHealth(
                component=component,
                status=ComponentStatus.OPERATIONAL,
                uptime_percentage=99.9,
                response_time_ms=150.0,
                error_rate=0.1
            )
    
    def get_system_status(self) -> SystemStatusPage:
        """Get overall system status page"""
        
        # Get all components
        components = list(components_store.values())
        
        # Determine overall status
        statuses = [c.status for c in components]
        if ComponentStatus.MAJOR_OUTAGE in statuses:
            overall_status = ComponentStatus.MAJOR_OUTAGE
        elif ComponentStatus.PARTIAL_OUTAGE in statuses:
            overall_status = ComponentStatus.PARTIAL_OUTAGE
        elif ComponentStatus.DEGRADED_PERFORMANCE in statuses:
            overall_status = ComponentStatus.DEGRADED_PERFORMANCE
        else:
            overall_status = ComponentStatus.OPERATIONAL
        
        # Get active incidents
        active_incidents = [
            i for i in incidents_store.values()
            if i.status != IncidentStatus.RESOLVED
        ]
        
        # Get recent incidents (last 7 days)
        week_ago = datetime.utcnow() - timedelta(days=7)
        recent_incidents = [
            i for i in incidents_store.values()
            if i.detected_at >= week_ago
        ]
        
        # Scheduled maintenance (mock data)
        scheduled_maintenance = [
            {
                "title": "Database Maintenance",
                "description": "Planned database upgrade",
                "scheduled_for": (datetime.utcnow() + timedelta(days=7)).isoformat(),
                "duration_minutes": 30,
                "affected_components": ["database"]
            }
        ]
        
        status_page = SystemStatusPage(
            overall_system_status=overall_status,
            active_incidents_count=len(active_incidents),
            components=components,
            active_incidents=active_incidents,
            uptime_7_days=99.9,
            uptime_30_days=99.8,
            uptime_90_days=99.7,
            recent_incidents=recent_incidents[:10],  # Last 10
            scheduled_maintenance=scheduled_maintenance
        )
        
        return status_page
    
    def get_tenant_view(self, tenant_id: str) -> TenantIncidentView:
        """Get tenant-specific incident view"""
        
        # Get incidents affecting this tenant
        tenant_incidents = [
            i for i in incidents_store.values()
            if (i.impact_scope == ImpactScope.ALL_TENANTS or
                tenant_id in i.affected_tenant_ids) and
                i.status != IncidentStatus.RESOLVED
        ]
        
        # Get component health
        components = list(components_store.values())
        
        # Determine overall status
        if any(i.severity == IncidentSeverity.CRITICAL for i in tenant_incidents):
            overall_status = ComponentStatus.MAJOR_OUTAGE
        elif any(i.severity == IncidentSeverity.HIGH for i in tenant_incidents):
            overall_status = ComponentStatus.PARTIAL_OUTAGE
        elif any(i.severity == IncidentSeverity.MEDIUM for i in tenant_incidents):
            overall_status = ComponentStatus.DEGRADED_PERFORMANCE
        else:
            overall_status = ComponentStatus.OPERATIONAL
        
        # Historical metrics (mock data)
        thirty_days_ago = datetime.utcnow() - timedelta(days=30)
        incidents_last_30_days = sum(
            1 for i in incidents_store.values()
            if (i.impact_scope == ImpactScope.ALL_TENANTS or tenant_id in i.affected_tenant_ids)
            and i.detected_at >= thirty_days_ago
        )
        
        # Calculate MTTR
        resolved_incidents = [
            i for i in incidents_store.values()
            if i.status == IncidentStatus.RESOLVED and
            i.time_to_resolve_minutes is not None
        ]
        mttr = sum(i.time_to_resolve_minutes for i in resolved_incidents) / len(resolved_incidents) if resolved_incidents else 0
        
        tenant_view = TenantIncidentView(
            tenant_id=tenant_id,
            tenant_name=f"Tenant {tenant_id}",
            overall_status=overall_status,
            active_incidents=tenant_incidents,
            component_health=components,
            incidents_last_30_days=incidents_last_30_days,
            uptime_last_30_days=99.8,
            mttr_minutes=mttr,
            sla_status="compliant",
            sla_breaches_this_month=0
        )
        
        tenant_views_store[tenant_id] = tenant_view
        return tenant_view
    
    async def create_incident(self, incident: Incident) -> Incident:
        """Create new incident"""
        
        incidents_store[incident.incident_id] = incident
        
        # Add initial update
        incident.updates.append({
            "timestamp": datetime.utcnow().isoformat(),
            "status": incident.status,
            "message": f"Incident detected: {incident.title}",
            "posted_by": "system"
        })
        
        # Update affected components
        for component in incident.affected_components:
            if component in components_store:
                health = components_store[component]
                if incident.severity == IncidentSeverity.CRITICAL:
                    health.status = ComponentStatus.MAJOR_OUTAGE
                elif incident.severity == IncidentSeverity.HIGH:
                    health.status = ComponentStatus.PARTIAL_OUTAGE
                else:
                    health.status = ComponentStatus.DEGRADED_PERFORMANCE
                
                health.last_incident = incident.incident_id
                health.last_incident_at = incident.detected_at
                health.affected_tenants = incident.affected_tenant_count
        
        # Notify subscribers
        await self._notify_subscribers(incident)
        
        # Broadcast via WebSocket
        await self._broadcast_incident_update(incident)
        
        self.logger.info(f"Created incident: {incident.title} (ID: {incident.incident_id})")
        
        return incident
    
    async def _notify_subscribers(self, incident: Incident):
        """Notify subscribers about incident"""
        
        # Get applicable subscriptions
        applicable_subs = [
            sub for sub in subscriptions_store.values()
            if sub.active and
            sub.min_severity.value <= incident.severity.value and
            (incident.impact_scope == ImpactScope.ALL_TENANTS or
             sub.tenant_id in incident.affected_tenant_ids)
        ]
        
        for sub in applicable_subs:
            notification = IncidentNotification(
                incident_id=incident.incident_id,
                tenant_id=sub.tenant_id,
                notification_type="email",
                recipient=sub.user_email,
                subject=f"[{incident.severity.value.upper()}] {incident.title}",
                message=f"{incident.description}\n\nStatus: {incident.status.value}\nAffected: {', '.join(c.value for c in incident.affected_components)}"
            )
            
            notifications_store[notification.notification_id] = notification
            incident.notifications_sent += 1
            incident.subscribers_notified.append(sub.user_email)
    
    async def _broadcast_incident_update(self, incident: Incident):
        """Broadcast incident update via WebSocket"""
        
        message = {
            "type": "incident_update",
            "incident_id": incident.incident_id,
            "title": incident.title,
            "severity": incident.severity,
            "status": incident.status,
            "affected_components": [c.value for c in incident.affected_components],
            "timestamp": datetime.utcnow().isoformat()
        }
        
        # Broadcast to all connected clients
        disconnected = set()
        for connection in active_connections:
            try:
                await connection.send_json(message)
            except:
                disconnected.add(connection)
        
        # Remove disconnected clients
        active_connections.difference_update(disconnected)

transparency_service = IncidentTransparencyService()

@app.get("/status", response_model=SystemStatusPage)
async def get_system_status():
    """Get public system status page"""
    return transparency_service.get_system_status()

@app.get("/status/tenant/{tenant_id}", response_model=TenantIncidentView)
async def get_tenant_status(tenant_id: str):
    """Get tenant-specific status view"""
    return transparency_service.get_tenant_view(tenant_id)

@app.get("/incidents", response_model=List[Incident])
async def list_incidents(
    tenant_id: Optional[str] = None,
    status: Optional[IncidentStatus] = None,
    severity: Optional[IncidentSeverity] = None,
    days: int = 30
):
    """List incidents with optional filters"""
    
    cutoff_date = datetime.utcnow() - timedelta(days=days)
    incidents = [i for i in incidents_store.values() if i.detected_at >= cutoff_date]
    
    if tenant_id:
        incidents = [
            i for i in incidents
            if i.impact_scope == ImpactScope.ALL_TENANTS or tenant_id in i.affected_tenant_ids
        ]
    
    if status:
        incidents = [i for i in incidents if i.status == status]
    
    if severity:
        incidents = [i for i in incidents if i.severity == severity]
    
    return incidents

@app.get("/incidents/{incident_id}", response_model=Incident)
async def get_incident(incident_id: str):
    """Get incident details"""
    
    if incident_id not in incidents_store:
        raise HTTPException(status_code=404, detail="Incident not found")
    
    return incidents_store[incident_id]

@app.post("/incidents", response_model=Incident)
async def create_incident(incident: Incident):
    """Create new incident (internal/admin only)"""
    
    try:
        created_incident = await transparency_service.create_incident(incident)
        return created_incident
        
    except Exception as e:
        logger.error(f"Failed to create incident: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/incidents/{incident_id}/update")
async def update_incident(
    incident_id: str,
    status: Optional[IncidentStatus] = None,
    update_message: str = ""
):
    """Post incident update"""
    
    if incident_id not in incidents_store:
        raise HTTPException(status_code=404, detail="Incident not found")
    
    incident = incidents_store[incident_id]
    
    # Add update
    incident.updates.append({
        "timestamp": datetime.utcnow().isoformat(),
        "status": status or incident.status,
        "message": update_message,
        "posted_by": "system"
    })
    
    # Update status if provided
    if status:
        old_status = incident.status
        incident.status = status
        
        if status == IncidentStatus.RESOLVED:
            incident.resolved_at = datetime.utcnow()
            
            # Calculate resolution time
            if incident.acknowledged_at:
                resolution_time = (incident.resolved_at - incident.acknowledged_at).total_seconds() / 60
                incident.time_to_resolve_minutes = resolution_time
            
            # Restore component health
            for component in incident.affected_components:
                if component in components_store:
                    components_store[component].status = ComponentStatus.OPERATIONAL
    
    # Broadcast update
    await transparency_service._broadcast_incident_update(incident)
    
    logger.info(f"Updated incident {incident_id}: {update_message}")
    
    return {
        "status": "updated",
        "incident_id": incident_id,
        "updated_at": datetime.utcnow().isoformat()
    }

@app.post("/subscriptions", response_model=IncidentSubscription)
async def create_subscription(subscription: IncidentSubscription):
    """Subscribe to incident notifications"""
    
    subscriptions_store[subscription.subscription_id] = subscription
    
    logger.info(f"Created subscription for {subscription.user_email}")
    
    return subscription

@app.get("/subscriptions/{tenant_id}", response_model=List[IncidentSubscription])
async def list_subscriptions(tenant_id: str):
    """List subscriptions for tenant"""
    
    subs = [s for s in subscriptions_store.values() if s.tenant_id == tenant_id]
    return subs

@app.delete("/subscriptions/{subscription_id}")
async def delete_subscription(subscription_id: str):
    """Unsubscribe from notifications"""
    
    if subscription_id in subscriptions_store:
        del subscriptions_store[subscription_id]
        return {"status": "unsubscribed", "subscription_id": subscription_id}
    
    raise HTTPException(status_code=404, detail="Subscription not found")

@app.websocket("/ws/status")
async def websocket_status_updates(websocket: WebSocket):
    """WebSocket endpoint for real-time status updates"""
    
    await websocket.accept()
    active_connections.add(websocket)
    
    try:
        # Send initial status
        status = transparency_service.get_system_status()
        await websocket.send_json({
            "type": "initial_status",
            "overall_status": status.overall_system_status,
            "active_incidents": len(status.active_incidents),
            "timestamp": datetime.utcnow().isoformat()
        })
        
        # Keep connection alive
        while True:
            data = await websocket.receive_text()
            # Echo back (keep-alive)
            await websocket.send_json({"type": "ping", "timestamp": datetime.utcnow().isoformat()})
            
    except WebSocketDisconnect:
        active_connections.remove(websocket)
        logger.info("WebSocket client disconnected")

@app.get("/metrics/uptime")
async def get_uptime_metrics(
    tenant_id: Optional[str] = None,
    days: int = 30
):
    """Get uptime metrics"""
    
    return {
        "period_days": days,
        "overall_uptime": 99.8,
        "by_component": {
            component.value: 99.9
            for component in ServiceComponent
        },
        "tenant_specific": tenant_id is not None
    }

@app.get("/metrics/incidents")
async def get_incident_metrics(days: int = 30):
    """Get incident metrics"""
    
    cutoff = datetime.utcnow() - timedelta(days=days)
    incidents = [i for i in incidents_store.values() if i.detected_at >= cutoff]
    
    return {
        "period_days": days,
        "total_incidents": len(incidents),
        "by_severity": {
            severity: sum(1 for i in incidents if i.severity == severity)
            for severity in IncidentSeverity
        },
        "resolved": sum(1 for i in incidents if i.status == IncidentStatus.RESOLVED),
        "average_resolution_time_minutes": sum(
            i.time_to_resolve_minutes for i in incidents if i.time_to_resolve_minutes
        ) / len(incidents) if incidents else 0
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8041)

