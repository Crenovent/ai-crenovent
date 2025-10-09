"""
Task 3.4.41: Build AI observability marketplace
- Plugin architecture for governance dashboards
- Marketplace backend with subscription management
- Dashboard discovery and installation system
- Multi-tenant plugin isolation and security
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from datetime import datetime
from enum import Enum
import uuid

app = FastAPI(title="RBIA AI Observability Marketplace")

class PluginCategory(str, Enum):
    GOVERNANCE = "governance"
    MONITORING = "monitoring"
    COMPLIANCE = "compliance"
    ANALYTICS = "analytics"
    VISUALIZATION = "visualization"

class PluginStatus(str, Enum):
    ACTIVE = "active"
    INACTIVE = "inactive"
    PENDING = "pending"
    DEPRECATED = "deprecated"

class MarketplacePlugin(BaseModel):
    plugin_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    plugin_name: str
    category: PluginCategory
    description: str
    version: str = "1.0.0"
    
    # Publisher info
    publisher_id: str
    publisher_name: str
    
    # Configuration
    config_schema: Dict[str, Any] = Field(default_factory=dict)
    default_config: Dict[str, Any] = Field(default_factory=dict)
    
    # Pricing
    pricing_model: str = "free"  # free, subscription, one_time
    price_per_month: float = 0.0
    
    # Compatibility
    supported_frameworks: List[str] = Field(default_factory=list)
    min_rbia_version: str = "1.0.0"
    
    # Metadata
    downloads: int = 0
    rating: float = 5.0
    status: PluginStatus = PluginStatus.ACTIVE
    
    created_at: datetime = Field(default_factory=datetime.utcnow)

class PluginSubscription(BaseModel):
    subscription_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    tenant_id: str
    plugin_id: str
    
    # Subscription details
    subscription_status: str = "active"  # active, suspended, cancelled
    installed_version: str
    custom_config: Dict[str, Any] = Field(default_factory=dict)
    
    # Billing
    billing_cycle: str = "monthly"
    next_billing_date: datetime
    
    # Usage
    last_used_at: Optional[datetime] = None
    usage_metrics: Dict[str, Any] = Field(default_factory=dict)
    
    subscribed_at: datetime = Field(default_factory=datetime.utcnow)

# Storage
plugins_store: Dict[str, MarketplacePlugin] = {}
subscriptions_store: Dict[str, PluginSubscription] = {}

def _initialize_default_plugins():
    """Initialize marketplace with default plugins"""
    
    plugins_store["governance_dashboard"] = MarketplacePlugin(
        plugin_id="governance_dashboard",
        plugin_name="Advanced Governance Dashboard",
        category=PluginCategory.GOVERNANCE,
        description="Comprehensive governance monitoring with policy enforcement tracking",
        publisher_id="rbia_official",
        publisher_name="RBIA Official",
        supported_frameworks=["SOX", "GDPR", "HIPAA"],
        rating=4.8,
        downloads=1250
    )
    
    plugins_store["compliance_reporter"] = MarketplacePlugin(
        plugin_id="compliance_reporter",
        plugin_name="Automated Compliance Reporter",
        category=PluginCategory.COMPLIANCE,
        description="Generate automated compliance reports for various frameworks",
        publisher_id="compliance_corp",
        publisher_name="Compliance Corp",
        pricing_model="subscription",
        price_per_month=99.0,
        supported_frameworks=["SOX", "GDPR", "RBI"],
        rating=4.6,
        downloads=890
    )

@app.on_event("startup")
async def startup_event():
    _initialize_default_plugins()

@app.get("/marketplace/plugins", response_model=List[MarketplacePlugin])
async def list_marketplace_plugins(
    category: Optional[PluginCategory] = None,
    search: Optional[str] = None,
    limit: int = 50
):
    """List available marketplace plugins"""
    
    plugins = list(plugins_store.values())
    
    if category:
        plugins = [p for p in plugins if p.category == category]
    
    if search:
        plugins = [p for p in plugins if search.lower() in p.plugin_name.lower() or search.lower() in p.description.lower()]
    
    # Sort by rating and downloads
    plugins.sort(key=lambda p: (p.rating, p.downloads), reverse=True)
    
    return plugins[:limit]

@app.get("/marketplace/plugins/{plugin_id}", response_model=MarketplacePlugin)
async def get_plugin_details(plugin_id: str):
    """Get detailed plugin information"""
    
    if plugin_id not in plugins_store:
        raise HTTPException(status_code=404, detail="Plugin not found")
    
    return plugins_store[plugin_id]

@app.post("/marketplace/subscribe", response_model=PluginSubscription)
async def subscribe_to_plugin(
    tenant_id: str,
    plugin_id: str,
    custom_config: Optional[Dict[str, Any]] = None
):
    """Subscribe tenant to a marketplace plugin"""
    
    if plugin_id not in plugins_store:
        raise HTTPException(status_code=404, detail="Plugin not found")
    
    plugin = plugins_store[plugin_id]
    
    # Check if already subscribed
    existing_sub = next(
        (s for s in subscriptions_store.values() 
         if s.tenant_id == tenant_id and s.plugin_id == plugin_id),
        None
    )
    
    if existing_sub:
        raise HTTPException(status_code=400, detail="Already subscribed to this plugin")
    
    subscription = PluginSubscription(
        tenant_id=tenant_id,
        plugin_id=plugin_id,
        installed_version=plugin.version,
        custom_config=custom_config or plugin.default_config,
        next_billing_date=datetime.utcnow().replace(day=1) if plugin.pricing_model == "subscription" else datetime.utcnow()
    )
    
    subscriptions_store[subscription.subscription_id] = subscription
    
    # Update plugin download count
    plugin.downloads += 1
    plugins_store[plugin_id] = plugin
    
    return subscription

@app.get("/marketplace/subscriptions/{tenant_id}", response_model=List[PluginSubscription])
async def get_tenant_subscriptions(tenant_id: str):
    """Get all plugin subscriptions for tenant"""
    
    return [s for s in subscriptions_store.values() if s.tenant_id == tenant_id]

@app.post("/marketplace/install/{subscription_id}")
async def install_plugin(subscription_id: str):
    """Install subscribed plugin for tenant"""
    
    if subscription_id not in subscriptions_store:
        raise HTTPException(status_code=404, detail="Subscription not found")
    
    subscription = subscriptions_store[subscription_id]
    plugin = plugins_store[subscription.plugin_id]
    
    # Simulate plugin installation
    installation_result = {
        "subscription_id": subscription_id,
        "plugin_name": plugin.plugin_name,
        "installation_status": "success",
        "dashboard_url": f"/dashboards/plugin/{plugin.plugin_id}",
        "config_applied": subscription.custom_config,
        "installed_at": datetime.utcnow().isoformat()
    }
    
    # Update subscription
    subscription.last_used_at = datetime.utcnow()
    subscriptions_store[subscription_id] = subscription
    
    return installation_result

@app.post("/marketplace/plugins", response_model=MarketplacePlugin)
async def publish_plugin(plugin: MarketplacePlugin):
    """Publish new plugin to marketplace (for plugin developers)"""
    
    plugin.status = PluginStatus.PENDING  # Requires approval
    plugins_store[plugin.plugin_id] = plugin
    
    return plugin

@app.get("/marketplace/analytics/{plugin_id}")
async def get_plugin_analytics(plugin_id: str):
    """Get analytics for plugin (for publishers)"""
    
    if plugin_id not in plugins_store:
        raise HTTPException(status_code=404, detail="Plugin not found")
    
    plugin = plugins_store[plugin_id]
    subscriptions = [s for s in subscriptions_store.values() if s.plugin_id == plugin_id]
    
    return {
        "plugin_id": plugin_id,
        "total_downloads": plugin.downloads,
        "active_subscriptions": len([s for s in subscriptions if s.subscription_status == "active"]),
        "monthly_revenue": sum(s.custom_config.get("monthly_fee", 0) for s in subscriptions),
        "average_rating": plugin.rating,
        "usage_trends": {
            "daily_active_users": len([s for s in subscriptions if s.last_used_at and (datetime.utcnow() - s.last_used_at).days < 1]),
            "weekly_active_users": len([s for s in subscriptions if s.last_used_at and (datetime.utcnow() - s.last_used_at).days < 7])
        }
    }

@app.get("/health")
async def health_check():
    return {"status": "healthy", "service": "AI Observability Marketplace", "task": "3.4.41"}
