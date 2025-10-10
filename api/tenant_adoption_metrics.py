"""
Task 4.4.17: Add adoption metric: number of tenants actively using RBIA templates
- Adoption growth metric
- Metrics ingestion from tenant usage data
- Board-level KPI
"""

from fastapi import FastAPI, HTTPException, BackgroundTasks
from pydantic import BaseModel, Field
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
from enum import Enum
import uuid
import logging
import asyncpg

app = FastAPI(title="RBIA Tenant Adoption Metrics")
logger = logging.getLogger(__name__)

class AdoptionStatus(str, Enum):
    NOT_STARTED = "not_started"
    TRIAL = "trial"
    ACTIVE = "active"
    CHURNED = "churned"

class UsageLevel(str, Enum):
    LOW = "low"        # < 10 executions/month
    MEDIUM = "medium"  # 10-100 executions/month
    HIGH = "high"      # > 100 executions/month

class TenantAdoptionMetric(BaseModel):
    metric_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    tenant_id: str
    tenant_name: str
    
    # Adoption status
    adoption_status: AdoptionStatus
    usage_level: UsageLevel
    
    # Usage statistics
    total_rbia_executions: int = 0
    total_rba_executions: int = 0
    rbia_adoption_percentage: float = 0.0
    
    # Template usage
    active_templates: List[str] = Field(default_factory=list)
    template_count: int = 0
    
    # Time metrics
    first_rbia_usage: Optional[datetime] = None
    last_rbia_usage: Optional[datetime] = None
    days_since_first_usage: int = 0
    
    # Engagement metrics
    monthly_active_users: int = 0
    weekly_executions: int = 0
    
    # Measurement period
    measurement_date: datetime = Field(default_factory=datetime.utcnow)
    measurement_period: str = "monthly"
    
    metadata: Dict[str, Any] = Field(default_factory=dict)

class AdoptionSummary(BaseModel):
    summary_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    
    # Overall metrics
    total_tenants: int
    active_rbia_tenants: int
    adoption_rate: float
    
    # Breakdown by status
    not_started_count: int
    trial_count: int
    active_count: int
    churned_count: int
    
    # Usage level breakdown
    low_usage_count: int
    medium_usage_count: int
    high_usage_count: int
    
    # Growth metrics
    new_adoptions_this_month: int
    churn_this_month: int
    net_growth: int
    
    # Template metrics
    total_template_executions: int
    avg_templates_per_tenant: float
    
    generated_at: datetime = Field(default_factory=datetime.utcnow)

# In-memory storage
adoption_metrics_store: Dict[str, TenantAdoptionMetric] = {}
adoption_summaries_store: Dict[str, AdoptionSummary] = {}

class TenantAdoptionTracker:
    def __init__(self):
        self.db_pool = None
        
    async def initialize(self):
        """Initialize database connection"""
        try:
            self.db_pool = await asyncpg.create_pool(
                host="localhost",
                port=5432,
                database="rbia_metrics",
                user="postgres",
                password="password",
                min_size=5,
                max_size=20
            )
            logger.info("✅ Tenant adoption tracker initialized")
            return True
        except Exception as e:
            logger.error(f"Failed to initialize tracker: {e}")
            return False
    
    async def collect_tenant_usage_data(self, tenant_id: str, days: int = 30) -> Dict[str, Any]:
        """Collect usage data for a tenant"""
        
        if not self.db_pool:
            return {}
            
        cutoff_date = datetime.utcnow() - timedelta(days=days)
        
        try:
            async with self.db_pool.acquire() as conn:
                # Get RBIA executions
                rbia_executions = await conn.fetchval("""
                    SELECT COUNT(*) FROM orchestrator_executions 
                    WHERE tenant_id = $1 AND workflow_type LIKE '%rbia%' 
                    AND created_at >= $2
                """, tenant_id, cutoff_date)
                
                # Get RBA executions
                rba_executions = await conn.fetchval("""
                    SELECT COUNT(*) FROM orchestrator_executions 
                    WHERE tenant_id = $1 AND workflow_type LIKE '%rba%' 
                    AND workflow_type NOT LIKE '%rbia%'
                    AND created_at >= $2
                """, tenant_id, cutoff_date)
                
                # Get active templates
                active_templates = await conn.fetch("""
                    SELECT DISTINCT template_id FROM rbia_traces 
                    WHERE tenant_id = $1 AND created_at >= $2
                """, tenant_id, cutoff_date)
                
                # Get first and last usage
                first_usage = await conn.fetchval("""
                    SELECT MIN(created_at) FROM rbia_traces 
                    WHERE tenant_id = $1
                """, tenant_id)
                
                last_usage = await conn.fetchval("""
                    SELECT MAX(created_at) FROM rbia_traces 
                    WHERE tenant_id = $1 AND created_at >= $2
                """, tenant_id, cutoff_date)
                
                # Get active users
                active_users = await conn.fetchval("""
                    SELECT COUNT(DISTINCT user_id) FROM rbia_traces 
                    WHERE tenant_id = $1 AND created_at >= $2
                """, tenant_id, cutoff_date)
                
                return {
                    "rbia_executions": rbia_executions or 0,
                    "rba_executions": rba_executions or 0,
                    "active_templates": [row['template_id'] for row in active_templates],
                    "first_usage": first_usage,
                    "last_usage": last_usage,
                    "active_users": active_users or 0
                }
                
        except Exception as e:
            logger.error(f"Failed to collect usage data for tenant {tenant_id}: {e}")
            return {}
    
    def determine_adoption_status(self, usage_data: Dict[str, Any]) -> AdoptionStatus:
        """Determine adoption status based on usage data"""
        
        rbia_executions = usage_data.get("rbia_executions", 0)
        last_usage = usage_data.get("last_usage")
        
        if rbia_executions == 0:
            return AdoptionStatus.NOT_STARTED
        
        # Check if churned (no usage in last 30 days)
        if last_usage and (datetime.utcnow() - last_usage).days > 30:
            return AdoptionStatus.CHURNED
        
        # Check if trial (< 50 executions total)
        if rbia_executions < 50:
            return AdoptionStatus.TRIAL
        
        return AdoptionStatus.ACTIVE
    
    def determine_usage_level(self, rbia_executions: int) -> UsageLevel:
        """Determine usage level based on execution count"""
        
        if rbia_executions < 10:
            return UsageLevel.LOW
        elif rbia_executions <= 100:
            return UsageLevel.MEDIUM
        else:
            return UsageLevel.HIGH
    
    async def calculate_tenant_metrics(self, tenant_id: str, tenant_name: str) -> TenantAdoptionMetric:
        """Calculate adoption metrics for a tenant"""
        
        # Collect usage data
        usage_data = await self.collect_tenant_usage_data(tenant_id)
        
        # Calculate metrics
        rbia_executions = usage_data.get("rbia_executions", 0)
        rba_executions = usage_data.get("rba_executions", 0)
        total_executions = rbia_executions + rba_executions
        
        rbia_adoption_percentage = (rbia_executions / total_executions * 100) if total_executions > 0 else 0.0
        
        adoption_status = self.determine_adoption_status(usage_data)
        usage_level = self.determine_usage_level(rbia_executions)
        
        # Calculate time metrics
        first_usage = usage_data.get("first_usage")
        days_since_first = (datetime.utcnow() - first_usage).days if first_usage else 0
        
        active_templates = usage_data.get("active_templates", [])
        
        return TenantAdoptionMetric(
            tenant_id=tenant_id,
            tenant_name=tenant_name,
            adoption_status=adoption_status,
            usage_level=usage_level,
            total_rbia_executions=rbia_executions,
            total_rba_executions=rba_executions,
            rbia_adoption_percentage=rbia_adoption_percentage,
            active_templates=active_templates,
            template_count=len(active_templates),
            first_rbia_usage=first_usage,
            last_rbia_usage=usage_data.get("last_usage"),
            days_since_first_usage=days_since_first,
            monthly_active_users=usage_data.get("active_users", 0),
            weekly_executions=rbia_executions // 4  # Approximate weekly from monthly
        )

# Global tracker instance
tracker = TenantAdoptionTracker()

@app.on_event("startup")
async def startup_event():
    await tracker.initialize()

@app.post("/adoption-metrics/calculate", response_model=TenantAdoptionMetric)
async def calculate_tenant_adoption(tenant_id: str, tenant_name: str):
    """Calculate adoption metrics for a tenant"""
    
    metric = await tracker.calculate_tenant_metrics(tenant_id, tenant_name)
    
    # Store metric
    adoption_metrics_store[metric.metric_id] = metric
    
    logger.info(f"✅ Calculated adoption metrics for tenant {tenant_name} - Status: {metric.adoption_status}")
    return metric

@app.get("/adoption-metrics/tenant/{tenant_id}")
async def get_tenant_adoption(tenant_id: str):
    """Get adoption metrics for a specific tenant"""
    
    tenant_metrics = [
        metric for metric in adoption_metrics_store.values()
        if metric.tenant_id == tenant_id
    ]
    
    if not tenant_metrics:
        raise HTTPException(status_code=404, detail="No adoption metrics found for tenant")
    
    # Return most recent metric
    latest_metric = max(tenant_metrics, key=lambda x: x.measurement_date)
    
    return latest_metric

@app.post("/adoption-metrics/summary", response_model=AdoptionSummary)
async def generate_adoption_summary(background_tasks: BackgroundTasks):
    """Generate overall adoption summary"""
    
    background_tasks.add_task(refresh_all_tenant_metrics)
    
    # Calculate summary from current metrics
    all_metrics = list(adoption_metrics_store.values())
    
    if not all_metrics:
        return AdoptionSummary(
            total_tenants=0,
            active_rbia_tenants=0,
            adoption_rate=0.0,
            not_started_count=0,
            trial_count=0,
            active_count=0,
            churned_count=0,
            low_usage_count=0,
            medium_usage_count=0,
            high_usage_count=0,
            new_adoptions_this_month=0,
            churn_this_month=0,
            net_growth=0,
            total_template_executions=0,
            avg_templates_per_tenant=0.0
        )
    
    # Get unique tenants (most recent metric per tenant)
    tenant_latest = {}
    for metric in all_metrics:
        if (metric.tenant_id not in tenant_latest or 
            metric.measurement_date > tenant_latest[metric.tenant_id].measurement_date):
            tenant_latest[metric.tenant_id] = metric
    
    latest_metrics = list(tenant_latest.values())
    
    total_tenants = len(latest_metrics)
    active_rbia_tenants = len([m for m in latest_metrics if m.adoption_status == AdoptionStatus.ACTIVE])
    adoption_rate = (active_rbia_tenants / total_tenants * 100) if total_tenants > 0 else 0.0
    
    # Status counts
    not_started_count = len([m for m in latest_metrics if m.adoption_status == AdoptionStatus.NOT_STARTED])
    trial_count = len([m for m in latest_metrics if m.adoption_status == AdoptionStatus.TRIAL])
    active_count = len([m for m in latest_metrics if m.adoption_status == AdoptionStatus.ACTIVE])
    churned_count = len([m for m in latest_metrics if m.adoption_status == AdoptionStatus.CHURNED])
    
    # Usage level counts
    low_usage_count = len([m for m in latest_metrics if m.usage_level == UsageLevel.LOW])
    medium_usage_count = len([m for m in latest_metrics if m.usage_level == UsageLevel.MEDIUM])
    high_usage_count = len([m for m in latest_metrics if m.usage_level == UsageLevel.HIGH])
    
    # Growth metrics (simplified - would need historical data)
    new_adoptions = len([m for m in latest_metrics if m.days_since_first_usage <= 30])
    churn_count = churned_count  # Simplified
    
    # Template metrics
    total_executions = sum([m.total_rbia_executions for m in latest_metrics])
    avg_templates = sum([m.template_count for m in latest_metrics]) / len(latest_metrics) if latest_metrics else 0.0
    
    summary = AdoptionSummary(
        total_tenants=total_tenants,
        active_rbia_tenants=active_rbia_tenants,
        adoption_rate=adoption_rate,
        not_started_count=not_started_count,
        trial_count=trial_count,
        active_count=active_count,
        churned_count=churned_count,
        low_usage_count=low_usage_count,
        medium_usage_count=medium_usage_count,
        high_usage_count=high_usage_count,
        new_adoptions_this_month=new_adoptions,
        churn_this_month=churn_count,
        net_growth=new_adoptions - churn_count,
        total_template_executions=total_executions,
        avg_templates_per_tenant=avg_templates
    )
    
    adoption_summaries_store[summary.summary_id] = summary
    
    logger.info(f"✅ Generated adoption summary - {active_rbia_tenants}/{total_tenants} tenants active ({adoption_rate:.1f}%)")
    return summary

async def refresh_all_tenant_metrics():
    """Background task to refresh metrics for all tenants"""
    
    # Get all unique tenants (would normally come from tenant registry)
    sample_tenants = [
        ("tenant_1", "Acme Corp"),
        ("tenant_2", "Beta Industries"),
        ("tenant_3", "Gamma Solutions")
    ]
    
    for tenant_id, tenant_name in sample_tenants:
        try:
            metric = await tracker.calculate_tenant_metrics(tenant_id, tenant_name)
            adoption_metrics_store[metric.metric_id] = metric
        except Exception as e:
            logger.error(f"Failed to refresh metrics for {tenant_id}: {e}")

@app.get("/adoption-metrics/board-kpis")
async def get_board_level_kpis():
    """Get board-level adoption KPIs"""
    
    # Get latest summary
    latest_summary = max(
        adoption_summaries_store.values(),
        key=lambda x: x.generated_at,
        default=None
    )
    
    if not latest_summary:
        return {
            "adoption_rate": 0.0,
            "total_active_tenants": 0,
            "monthly_growth": 0,
            "churn_rate": 0.0
        }
    
    churn_rate = (latest_summary.churn_this_month / latest_summary.total_tenants * 100) if latest_summary.total_tenants > 0 else 0.0
    
    return {
        "adoption_rate": latest_summary.adoption_rate,
        "total_active_tenants": latest_summary.active_rbia_tenants,
        "monthly_growth": latest_summary.net_growth,
        "churn_rate": churn_rate,
        "total_template_executions": latest_summary.total_template_executions
    }

@app.get("/adoption-metrics/trends")
async def get_adoption_trends(days: int = 90):
    """Get adoption trends over time"""
    
    cutoff_date = datetime.utcnow() - timedelta(days=days)
    
    # Get historical summaries
    historical_summaries = [
        summary for summary in adoption_summaries_store.values()
        if summary.generated_at >= cutoff_date
    ]
    
    # Sort by date
    historical_summaries.sort(key=lambda x: x.generated_at)
    
    trends = []
    for summary in historical_summaries:
        trends.append({
            "date": summary.generated_at.isoformat(),
            "adoption_rate": summary.adoption_rate,
            "active_tenants": summary.active_rbia_tenants,
            "total_tenants": summary.total_tenants,
            "new_adoptions": summary.new_adoptions_this_month
        })
    
    return {
        "trend_data": trends,
        "trend_count": len(trends)
    }
