"""
Task 3.2.29: Cost guardrails (FinOps budgets per tenant/model, alerting, auto throttle)
- Cost predictability
- Metering + policies
- Reports to CFO persona
"""

from fastapi import FastAPI, HTTPException, BackgroundTasks
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any, Union
from datetime import datetime, timedelta
from enum import Enum
import uuid
import logging
import json
import asyncio
from dataclasses import dataclass
import statistics

app = FastAPI(title="RBIA FinOps Cost Guardrails Service")
logger = logging.getLogger(__name__)

class BudgetPeriod(str, Enum):
    DAILY = "daily"
    WEEKLY = "weekly"
    MONTHLY = "monthly"
    QUARTERLY = "quarterly"
    YEARLY = "yearly"

class CostCategory(str, Enum):
    COMPUTE = "compute"
    STORAGE = "storage"
    NETWORK = "network"
    ML_INFERENCE = "ml_inference"
    DATA_PROCESSING = "data_processing"
    EXTERNAL_APIS = "external_apis"
    TOTAL = "total"

class AlertSeverity(str, Enum):
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"
    EMERGENCY = "emergency"

class ThrottleAction(str, Enum):
    NONE = "none"
    RATE_LIMIT = "rate_limit"
    QUEUE_REQUESTS = "queue_requests"
    BLOCK_NEW_REQUESTS = "block_new_requests"
    EMERGENCY_SHUTDOWN = "emergency_shutdown"

class BudgetStatus(str, Enum):
    HEALTHY = "healthy"
    WARNING = "warning"  # 80% threshold
    CRITICAL = "critical"  # 95% threshold
    EXCEEDED = "exceeded"  # 100%+ threshold
    EMERGENCY = "emergency"  # 120%+ threshold

class TenantTier(str, Enum):
    T0_REGULATED = "T0"
    T1_ENTERPRISE = "T1"
    T2_MIDMARKET = "T2"

class CostBudget(BaseModel):
    budget_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    tenant_id: str
    model_id: Optional[str] = None  # None means tenant-wide budget
    
    # Budget configuration
    budget_name: str
    budget_period: BudgetPeriod
    budget_amount_usd: float = Field(..., gt=0)
    
    # Category breakdown (optional)
    category_limits: Dict[CostCategory, float] = Field(default_factory=dict)
    
    # Thresholds and actions
    warning_threshold_percent: float = Field(default=80.0, ge=0, le=100)
    critical_threshold_percent: float = Field(default=95.0, ge=0, le=100)
    emergency_threshold_percent: float = Field(default=120.0, ge=100)
    
    # Throttling configuration
    throttle_at_warning: ThrottleAction = ThrottleAction.NONE
    throttle_at_critical: ThrottleAction = ThrottleAction.RATE_LIMIT
    throttle_at_exceeded: ThrottleAction = ThrottleAction.BLOCK_NEW_REQUESTS
    throttle_at_emergency: ThrottleAction = ThrottleAction.EMERGENCY_SHUTDOWN
    
    # Metadata
    created_at: datetime = Field(default_factory=datetime.utcnow)
    created_by: str
    active: bool = True
    
    # Current period tracking
    current_period_start: datetime = Field(default_factory=datetime.utcnow)
    current_period_end: datetime = Field(default_factory=lambda: datetime.utcnow() + timedelta(days=30))

class CostUsage(BaseModel):
    usage_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    tenant_id: str
    model_id: Optional[str] = None
    
    # Cost breakdown
    cost_category: CostCategory
    cost_amount_usd: float = Field(..., ge=0)
    
    # Usage details
    resource_type: str  # e.g., "inference_request", "storage_gb_hour", "compute_hour"
    resource_quantity: float
    unit_cost_usd: float
    
    # Context
    workflow_id: Optional[str] = None
    user_id: Optional[str] = None
    region: Optional[str] = None
    
    # Metadata
    recorded_by: str = "cost_metering_service"
    metadata: Dict[str, Any] = Field(default_factory=dict)

class CostAlert(BaseModel):
    alert_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    tenant_id: str
    budget_id: str
    
    # Alert details
    severity: AlertSeverity
    message: str
    current_spend_usd: float
    budget_amount_usd: float
    utilization_percent: float
    
    # Projections
    projected_monthly_spend_usd: Optional[float] = None
    days_until_budget_exceeded: Optional[int] = None
    
    # Actions taken
    throttle_action_applied: ThrottleAction = ThrottleAction.NONE
    notification_sent: bool = False
    
    # Recipients
    notify_users: List[str] = Field(default_factory=list)
    notify_roles: List[str] = Field(default_factory=lambda: ["cfo", "finops_manager"])

class ThrottlePolicy(BaseModel):
    policy_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    tenant_id: str
    model_id: Optional[str] = None
    
    # Throttling rules
    max_requests_per_minute: Optional[int] = None
    max_concurrent_requests: Optional[int] = None
    queue_max_size: Optional[int] = None
    queue_timeout_seconds: Optional[int] = None
    
    # Cost-based throttling
    cost_per_request_limit_usd: Optional[float] = None
    daily_cost_limit_usd: Optional[float] = None
    
    # Status
    active: bool = True
    applied_at: Optional[datetime] = None
    applied_reason: Optional[str] = None

class FinOpsReport(BaseModel):
    report_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    generated_at: datetime = Field(default_factory=datetime.utcnow)
    report_period_start: datetime
    report_period_end: datetime
    
    # Summary metrics
    total_tenants: int
    total_spend_usd: float
    average_spend_per_tenant_usd: float
    
    # Budget performance
    budgets_healthy: int
    budgets_warning: int
    budgets_critical: int
    budgets_exceeded: int
    
    # Top spenders
    top_spending_tenants: List[Dict[str, Any]] = Field(default_factory=list)
    top_spending_models: List[Dict[str, Any]] = Field(default_factory=list)
    
    # Cost breakdown
    cost_by_category: Dict[CostCategory, float] = Field(default_factory=dict)
    cost_by_tenant_tier: Dict[TenantTier, float] = Field(default_factory=dict)
    
    # Trends
    month_over_month_change_percent: Optional[float] = None
    cost_efficiency_metrics: Dict[str, float] = Field(default_factory=dict)
    
    # Recommendations
    cost_optimization_recommendations: List[str] = Field(default_factory=list)

# In-memory stores (replace with proper database in production)
cost_budgets: Dict[str, CostBudget] = {}
cost_usage: List[CostUsage] = []
cost_alerts: List[CostAlert] = []
throttle_policies: Dict[str, ThrottlePolicy] = {}

# Current usage tracking per tenant/model
current_usage: Dict[str, Dict[CostCategory, float]] = {}  # {tenant_id: {category: amount}}

def _get_budget_key(tenant_id: str, model_id: Optional[str] = None) -> str:
    """Generate a key for budget lookups."""
    return f"{tenant_id}:{model_id or 'tenant_wide'}"

def _calculate_current_spend(tenant_id: str, model_id: Optional[str] = None, 
                           period_start: datetime = None, period_end: datetime = None) -> Dict[CostCategory, float]:
    """Calculate current spend for a tenant/model within a period."""
    if not period_start:
        period_start = datetime.utcnow() - timedelta(days=30)
    if not period_end:
        period_end = datetime.utcnow()
    
    filtered_usage = [
        u for u in cost_usage
        if u.tenant_id == tenant_id
        and (model_id is None or u.model_id == model_id)
        and period_start <= u.timestamp <= period_end
    ]
    
    spend_by_category = {}
    for category in CostCategory:
        category_spend = sum(u.cost_amount_usd for u in filtered_usage if u.cost_category == category)
        spend_by_category[category] = category_spend
    
    return spend_by_category

def _determine_budget_status(current_spend: float, budget_amount: float, 
                           warning_threshold: float, critical_threshold: float,
                           emergency_threshold: float) -> BudgetStatus:
    """Determine budget status based on current spend and thresholds."""
    utilization_percent = (current_spend / budget_amount) * 100
    
    if utilization_percent >= emergency_threshold:
        return BudgetStatus.EMERGENCY
    elif utilization_percent >= 100:
        return BudgetStatus.EXCEEDED
    elif utilization_percent >= critical_threshold:
        return BudgetStatus.CRITICAL
    elif utilization_percent >= warning_threshold:
        return BudgetStatus.WARNING
    else:
        return BudgetStatus.HEALTHY

async def _apply_throttle_policy(tenant_id: str, model_id: Optional[str], 
                               action: ThrottleAction, reason: str) -> ThrottlePolicy:
    """Apply throttling policy based on budget status."""
    policy_key = _get_budget_key(tenant_id, model_id)
    
    if action == ThrottleAction.NONE:
        # Remove any existing throttle
        if policy_key in throttle_policies:
            throttle_policies[policy_key].active = False
        return None
    
    # Create or update throttle policy
    throttle_config = {
        ThrottleAction.RATE_LIMIT: {
            "max_requests_per_minute": 60,
            "max_concurrent_requests": 10
        },
        ThrottleAction.QUEUE_REQUESTS: {
            "max_requests_per_minute": 30,
            "max_concurrent_requests": 5,
            "queue_max_size": 100,
            "queue_timeout_seconds": 300
        },
        ThrottleAction.BLOCK_NEW_REQUESTS: {
            "max_requests_per_minute": 0,
            "max_concurrent_requests": 0
        },
        ThrottleAction.EMERGENCY_SHUTDOWN: {
            "max_requests_per_minute": 0,
            "max_concurrent_requests": 0,
            "daily_cost_limit_usd": 0
        }
    }
    
    config = throttle_config.get(action, {})
    
    policy = ThrottlePolicy(
        tenant_id=tenant_id,
        model_id=model_id,
        applied_at=datetime.utcnow(),
        applied_reason=reason,
        **config
    )
    
    throttle_policies[policy_key] = policy
    logger.warning(f"Applied throttle policy {action.value} for tenant {tenant_id}, model {model_id}: {reason}")
    
    return policy

async def _send_cost_alert(budget: CostBudget, current_spend: float, 
                         severity: AlertSeverity, background_tasks: BackgroundTasks):
    """Send cost alert to stakeholders."""
    utilization_percent = (current_spend / budget.budget_amount_usd) * 100
    
    # Calculate projections
    days_in_period = (budget.current_period_end - budget.current_period_start).days
    days_elapsed = (datetime.utcnow() - budget.current_period_start).days
    
    if days_elapsed > 0:
        daily_burn_rate = current_spend / days_elapsed
        projected_monthly_spend = daily_burn_rate * days_in_period
        days_until_exceeded = max(0, (budget.budget_amount_usd - current_spend) / daily_burn_rate) if daily_burn_rate > 0 else None
    else:
        projected_monthly_spend = None
        days_until_exceeded = None
    
    alert = CostAlert(
        tenant_id=budget.tenant_id,
        budget_id=budget.budget_id,
        severity=severity,
        message=f"Budget '{budget.budget_name}' is at {utilization_percent:.1f}% utilization (${current_spend:.2f} of ${budget.budget_amount_usd:.2f})",
        current_spend_usd=current_spend,
        budget_amount_usd=budget.budget_amount_usd,
        utilization_percent=utilization_percent,
        projected_monthly_spend_usd=projected_monthly_spend,
        days_until_budget_exceeded=int(days_until_exceeded) if days_until_exceeded else None
    )
    
    cost_alerts.append(alert)
    
    # In production, this would send actual notifications (email, Slack, etc.)
    logger.warning(f"Cost alert generated: {alert.message}")
    
    return alert

@app.post("/finops/budgets", response_model=CostBudget)
async def create_cost_budget(budget: CostBudget):
    """Create a new cost budget for a tenant or specific model."""
    
    # Validate budget configuration
    if budget.critical_threshold_percent <= budget.warning_threshold_percent:
        raise HTTPException(status_code=400, detail="Critical threshold must be higher than warning threshold")
    
    # Set period end based on budget period
    period_delta = {
        BudgetPeriod.DAILY: timedelta(days=1),
        BudgetPeriod.WEEKLY: timedelta(weeks=1),
        BudgetPeriod.MONTHLY: timedelta(days=30),
        BudgetPeriod.QUARTERLY: timedelta(days=90),
        BudgetPeriod.YEARLY: timedelta(days=365)
    }
    
    budget.current_period_end = budget.current_period_start + period_delta[budget.budget_period]
    
    cost_budgets[budget.budget_id] = budget
    
    logger.info(f"Created cost budget {budget.budget_id} for tenant {budget.tenant_id}, model {budget.model_id}: ${budget.budget_amount_usd}")
    return budget

@app.post("/finops/usage", response_model=CostUsage)
async def record_cost_usage(
    usage: CostUsage,
    background_tasks: BackgroundTasks,
    check_budgets: bool = True
):
    """Record cost usage and check against budgets."""
    
    cost_usage.append(usage)
    
    # Update current usage tracking
    tenant_key = usage.tenant_id
    if tenant_key not in current_usage:
        current_usage[tenant_key] = {category: 0.0 for category in CostCategory}
    
    current_usage[tenant_key][usage.cost_category] += usage.cost_amount_usd
    current_usage[tenant_key][CostCategory.TOTAL] += usage.cost_amount_usd
    
    if check_budgets:
        # Check budgets in background
        background_tasks.add_task(_check_budgets_for_tenant, usage.tenant_id, usage.model_id)
    
    logger.debug(f"Recorded cost usage: {usage.tenant_id} - {usage.cost_category.value}: ${usage.cost_amount_usd}")
    return usage

async def _check_budgets_for_tenant(tenant_id: str, model_id: Optional[str] = None):
    """Check budgets for a tenant and apply throttling if needed."""
    
    # Find applicable budgets
    applicable_budgets = [
        b for b in cost_budgets.values()
        if b.tenant_id == tenant_id and b.active
        and (b.model_id is None or b.model_id == model_id)
    ]
    
    for budget in applicable_budgets:
        current_spend_by_category = _calculate_current_spend(
            tenant_id, budget.model_id, budget.current_period_start, budget.current_period_end
        )
        
        total_spend = current_spend_by_category[CostCategory.TOTAL]
        status = _determine_budget_status(
            total_spend, budget.budget_amount_usd,
            budget.warning_threshold_percent, budget.critical_threshold_percent,
            budget.emergency_threshold_percent
        )
        
        # Determine appropriate actions
        if status == BudgetStatus.EMERGENCY:
            # Check for budget exhaustion (100% utilization) - Task 6.4.14
            utilization = (total_spend / budget.budget_amount_usd) * 100
            if utilization >= 100.0:
                await _trigger_budget_exhaustion_fallback(tenant_id, budget, total_spend)
            
            await _apply_throttle_policy(tenant_id, budget.model_id, budget.throttle_at_emergency, 
                                       f"Emergency budget threshold exceeded: {total_spend:.2f} / {budget.budget_amount_usd:.2f}")
            await _send_cost_alert(budget, total_spend, AlertSeverity.EMERGENCY, None)
        elif status == BudgetStatus.EXCEEDED:
            await _apply_throttle_policy(tenant_id, budget.model_id, budget.throttle_at_exceeded,
                                       f"Budget exceeded: {total_spend:.2f} / {budget.budget_amount_usd:.2f}")
            await _send_cost_alert(budget, total_spend, AlertSeverity.CRITICAL, None)
        elif status == BudgetStatus.CRITICAL:
            await _apply_throttle_policy(tenant_id, budget.model_id, budget.throttle_at_critical,
                                       f"Critical budget threshold reached: {total_spend:.2f} / {budget.budget_amount_usd:.2f}")
            await _send_cost_alert(budget, total_spend, AlertSeverity.CRITICAL, None)
        elif status == BudgetStatus.WARNING:
            await _apply_throttle_policy(tenant_id, budget.model_id, budget.throttle_at_warning,
                                       f"Warning budget threshold reached: {total_spend:.2f} / {budget.budget_amount_usd:.2f}")
            await _send_cost_alert(budget, total_spend, AlertSeverity.WARNING, None)

@app.get("/finops/budgets/{tenant_id}", response_model=List[CostBudget])
async def get_tenant_budgets(tenant_id: str, model_id: Optional[str] = None):
    """Get all budgets for a tenant."""
    
    budgets = [
        b for b in cost_budgets.values()
        if b.tenant_id == tenant_id
        and (model_id is None or b.model_id == model_id)
    ]
    
    return budgets

@app.get("/finops/usage/{tenant_id}")
async def get_tenant_usage(
    tenant_id: str,
    model_id: Optional[str] = None,
    start_date: Optional[datetime] = None,
    end_date: Optional[datetime] = None
):
    """Get cost usage for a tenant with optional filters."""
    
    if not start_date:
        start_date = datetime.utcnow() - timedelta(days=30)
    if not end_date:
        end_date = datetime.utcnow()
    
    filtered_usage = [
        u for u in cost_usage
        if u.tenant_id == tenant_id
        and (model_id is None or u.model_id == model_id)
        and start_date <= u.timestamp <= end_date
    ]
    
    # Aggregate by category
    usage_by_category = {}
    for category in CostCategory:
        category_usage = [u for u in filtered_usage if u.cost_category == category]
        usage_by_category[category.value] = {
            "total_cost_usd": sum(u.cost_amount_usd for u in category_usage),
            "total_quantity": sum(u.resource_quantity for u in category_usage),
            "usage_count": len(category_usage)
        }
    
    return {
        "tenant_id": tenant_id,
        "model_id": model_id,
        "period_start": start_date,
        "period_end": end_date,
        "usage_by_category": usage_by_category,
        "total_cost_usd": sum(u.cost_amount_usd for u in filtered_usage)
    }

@app.get("/finops/budget-status/{tenant_id}")
async def get_budget_status(tenant_id: str, model_id: Optional[str] = None):
    """Get current budget status for a tenant."""
    
    applicable_budgets = [
        b for b in cost_budgets.values()
        if b.tenant_id == tenant_id and b.active
        and (model_id is None or b.model_id == model_id)
    ]
    
    budget_statuses = []
    
    for budget in applicable_budgets:
        current_spend_by_category = _calculate_current_spend(
            tenant_id, budget.model_id, budget.current_period_start, budget.current_period_end
        )
        
        total_spend = current_spend_by_category[CostCategory.TOTAL]
        status = _determine_budget_status(
            total_spend, budget.budget_amount_usd,
            budget.warning_threshold_percent, budget.critical_threshold_percent,
            budget.emergency_threshold_percent
        )
        
        utilization_percent = (total_spend / budget.budget_amount_usd) * 100
        
        budget_statuses.append({
            "budget_id": budget.budget_id,
            "budget_name": budget.budget_name,
            "budget_amount_usd": budget.budget_amount_usd,
            "current_spend_usd": total_spend,
            "utilization_percent": utilization_percent,
            "status": status.value,
            "spend_by_category": current_spend_by_category,
            "period_start": budget.current_period_start,
            "period_end": budget.current_period_end
        })
    
    return {
        "tenant_id": tenant_id,
        "model_id": model_id,
        "budget_statuses": budget_statuses
    }

@app.get("/finops/alerts", response_model=List[CostAlert])
async def get_cost_alerts(
    tenant_id: Optional[str] = None,
    severity: Optional[AlertSeverity] = None,
    limit: int = 100
):
    """Get cost alerts with optional filters."""
    
    filtered_alerts = cost_alerts
    
    if tenant_id:
        filtered_alerts = [a for a in filtered_alerts if a.tenant_id == tenant_id]
    
    if severity:
        filtered_alerts = [a for a in filtered_alerts if a.severity == severity]
    
    # Sort by timestamp (most recent first)
    filtered_alerts.sort(key=lambda x: x.timestamp, reverse=True)
    
    return filtered_alerts[:limit]

@app.get("/finops/throttle-policies/{tenant_id}")
async def get_throttle_policies(tenant_id: str, model_id: Optional[str] = None):
    """Get active throttle policies for a tenant."""
    
    policies = [
        p for p in throttle_policies.values()
        if p.tenant_id == tenant_id and p.active
        and (model_id is None or p.model_id == model_id)
    ]
    
    return policies

@app.post("/finops/check-request-allowed")
async def check_request_allowed(
    tenant_id: str,
    model_id: Optional[str] = None,
    estimated_cost_usd: Optional[float] = None
):
    """Check if a request is allowed based on current throttle policies and budgets."""
    
    policy_key = _get_budget_key(tenant_id, model_id)
    
    # Check throttle policies
    if policy_key in throttle_policies:
        policy = throttle_policies[policy_key]
        if policy.active:
            if policy.max_requests_per_minute == 0:
                return {
                    "allowed": False,
                    "reason": "Requests blocked due to budget constraints",
                    "throttle_action": policy.applied_reason
                }
    
    # Check budget constraints
    if estimated_cost_usd:
        applicable_budgets = [
            b for b in cost_budgets.values()
            if b.tenant_id == tenant_id and b.active
            and (b.model_id is None or b.model_id == model_id)
        ]
        
        for budget in applicable_budgets:
            current_spend_by_category = _calculate_current_spend(
                tenant_id, budget.model_id, budget.current_period_start, budget.current_period_end
            )
            
            total_spend = current_spend_by_category[CostCategory.TOTAL]
            projected_spend = total_spend + estimated_cost_usd
            
            if projected_spend > budget.budget_amount_usd:
                return {
                    "allowed": False,
                    "reason": f"Request would exceed budget: ${projected_spend:.2f} > ${budget.budget_amount_usd:.2f}",
                    "current_spend": total_spend,
                    "budget_amount": budget.budget_amount_usd
                }
    
    return {"allowed": True, "reason": "Request approved"}

@app.get("/finops/reports/cfo-summary", response_model=FinOpsReport)
async def generate_cfo_report(
    start_date: Optional[datetime] = None,
    end_date: Optional[datetime] = None
):
    """Generate CFO-facing FinOps summary report."""
    
    if not start_date:
        start_date = datetime.utcnow() - timedelta(days=30)
    if not end_date:
        end_date = datetime.utcnow()
    
    # Filter usage for the period
    period_usage = [
        u for u in cost_usage
        if start_date <= u.timestamp <= end_date
    ]
    
    # Calculate summary metrics
    total_spend = sum(u.cost_amount_usd for u in period_usage)
    unique_tenants = len(set(u.tenant_id for u in period_usage))
    avg_spend_per_tenant = total_spend / unique_tenants if unique_tenants > 0 else 0
    
    # Budget performance
    budget_statuses = {"healthy": 0, "warning": 0, "critical": 0, "exceeded": 0}
    for budget in cost_budgets.values():
        if budget.active:
            current_spend_by_category = _calculate_current_spend(
                budget.tenant_id, budget.model_id, budget.current_period_start, budget.current_period_end
            )
            total_spend_budget = current_spend_by_category[CostCategory.TOTAL]
            status = _determine_budget_status(
                total_spend_budget, budget.budget_amount_usd,
                budget.warning_threshold_percent, budget.critical_threshold_percent,
                budget.emergency_threshold_percent
            )
            
            if status == BudgetStatus.HEALTHY:
                budget_statuses["healthy"] += 1
            elif status == BudgetStatus.WARNING:
                budget_statuses["warning"] += 1
            elif status in [BudgetStatus.CRITICAL, BudgetStatus.EMERGENCY]:
                budget_statuses["critical"] += 1
            elif status == BudgetStatus.EXCEEDED:
                budget_statuses["exceeded"] += 1
    
    # Top spenders
    tenant_spend = {}
    model_spend = {}
    
    for usage in period_usage:
        tenant_spend[usage.tenant_id] = tenant_spend.get(usage.tenant_id, 0) + usage.cost_amount_usd
        if usage.model_id:
            model_key = f"{usage.tenant_id}:{usage.model_id}"
            model_spend[model_key] = model_spend.get(model_key, 0) + usage.cost_amount_usd
    
    top_tenants = [
        {"tenant_id": tid, "spend_usd": spend}
        for tid, spend in sorted(tenant_spend.items(), key=lambda x: x[1], reverse=True)[:10]
    ]
    
    top_models = [
        {"model_key": mid, "spend_usd": spend}
        for mid, spend in sorted(model_spend.items(), key=lambda x: x[1], reverse=True)[:10]
    ]
    
    # Cost by category
    cost_by_category = {}
    for category in CostCategory:
        if category != CostCategory.TOTAL:
            category_spend = sum(u.cost_amount_usd for u in period_usage if u.cost_category == category)
            cost_by_category[category] = category_spend
    
    # Generate recommendations
    recommendations = []
    if budget_statuses["exceeded"] > 0:
        recommendations.append(f"{budget_statuses['exceeded']} budgets are exceeded - consider increasing limits or optimizing usage")
    if budget_statuses["critical"] > 0:
        recommendations.append(f"{budget_statuses['critical']} budgets are in critical state - immediate attention required")
    if avg_spend_per_tenant > 1000:
        recommendations.append("High average spend per tenant - consider cost optimization initiatives")
    
    report = FinOpsReport(
        report_period_start=start_date,
        report_period_end=end_date,
        total_tenants=unique_tenants,
        total_spend_usd=total_spend,
        average_spend_per_tenant_usd=avg_spend_per_tenant,
        budgets_healthy=budget_statuses["healthy"],
        budgets_warning=budget_statuses["warning"],
        budgets_critical=budget_statuses["critical"],
        budgets_exceeded=budget_statuses["exceeded"],
        top_spending_tenants=top_tenants,
        top_spending_models=top_models,
        cost_by_category=cost_by_category,
        cost_optimization_recommendations=recommendations
    )
    
    return report

@app.post("/finops/budgets/{budget_id}/reset-period")
async def reset_budget_period(budget_id: str):
    """Reset a budget period (typically done at the start of a new period)."""
    
    if budget_id not in cost_budgets:
        raise HTTPException(status_code=404, detail="Budget not found")
    
    budget = cost_budgets[budget_id]
    
    # Calculate new period
    period_delta = {
        BudgetPeriod.DAILY: timedelta(days=1),
        BudgetPeriod.WEEKLY: timedelta(weeks=1),
        BudgetPeriod.MONTHLY: timedelta(days=30),
        BudgetPeriod.QUARTERLY: timedelta(days=90),
        BudgetPeriod.YEARLY: timedelta(days=365)
    }
    
    budget.current_period_start = datetime.utcnow()
    budget.current_period_end = budget.current_period_start + period_delta[budget.budget_period]
    
    # Remove any throttle policies
    policy_key = _get_budget_key(budget.tenant_id, budget.model_id)
    if policy_key in throttle_policies:
        throttle_policies[policy_key].active = False
    
    logger.info(f"Reset budget period for {budget_id}: {budget.current_period_start} to {budget.current_period_end}")
    
    return {
        "budget_id": budget_id,
        "new_period_start": budget.current_period_start,
        "new_period_end": budget.current_period_end,
        "throttle_policies_cleared": True
    }

async def _trigger_budget_exhaustion_fallback(
    tenant_id: str, 
    budget: 'CostBudget', 
    current_spend: float
):
    """Trigger fallback when budget is exhausted - Task 6.4.14"""
    try:
        fallback_data = {
            "request_id": f"budget_exhaustion_{uuid.uuid4()}",
            "tenant_id": tenant_id,
            "workflow_id": "cost_management",
            "current_system": "rbia",
            "error_type": "budget_exhaustion",
            "error_message": f"Budget exhausted: ${current_spend:.2f} / ${budget.budget_amount_usd:.2f}",
            "budget_id": budget.budget_id,
            "utilization_percent": (current_spend / budget.budget_amount_usd) * 100
        }
        
        logger.warning(f"Triggering budget exhaustion fallback: {json.dumps(fallback_data)}")
        
    except Exception as fallback_error:
        logger.error(f"Failed to trigger budget exhaustion fallback: {fallback_error}")

@app.get("/health")
async def health_check():
    return {"status": "healthy", "service": "RBIA FinOps Cost Guardrails Service", "task": "3.2.29"}
