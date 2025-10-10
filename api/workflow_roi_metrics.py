"""
Task 4.4.42: Add workflow ROI metric (revenue impact per intelligent template)
- Business alignment
- BI dashboards
- CRO persona
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
from enum import Enum
import uuid
import logging

app = FastAPI(title="RBIA Workflow ROI Metrics")
logger = logging.getLogger(__name__)

class WorkflowCategory(str, Enum):
    SALES = "sales"
    MARKETING = "marketing"
    OPERATIONS = "operations"
    FINANCE = "finance"
    CUSTOMER_SUCCESS = "customer_success"

class ROIMetricType(str, Enum):
    REVENUE_INCREASE = "revenue_increase"
    COST_REDUCTION = "cost_reduction"
    TIME_SAVINGS = "time_savings"
    EFFICIENCY_GAIN = "efficiency_gain"

class WorkflowROIMetric(BaseModel):
    metric_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    tenant_id: str
    
    # Workflow identification
    workflow_id: str
    workflow_name: str
    template_id: str
    template_name: str
    workflow_category: WorkflowCategory
    
    # ROI calculations
    revenue_impact: float = 0.0
    cost_savings: float = 0.0
    time_saved_hours: float = 0.0
    efficiency_percentage: float = 0.0
    
    # Investment metrics
    implementation_cost: float = 0.0
    operational_cost_monthly: float = 0.0
    total_investment: float = 0.0
    
    # ROI calculations
    total_benefit: float = 0.0
    net_benefit: float = 0.0
    roi_percentage: float = 0.0
    payback_months: float = 0.0
    
    # Usage metrics
    executions_count: int = 0
    active_users: int = 0
    success_rate: float = 0.0
    
    # Per-execution metrics
    revenue_per_execution: float = 0.0
    cost_per_execution: float = 0.0
    time_saved_per_execution: float = 0.0
    
    # Time period
    measurement_date: datetime = Field(default_factory=datetime.utcnow)
    measurement_period: str = "monthly"
    
    metadata: Dict[str, Any] = Field(default_factory=dict)

class WorkflowUsageData(BaseModel):
    workflow_id: str
    executions: int
    successful_executions: int
    active_users: int
    avg_execution_time_minutes: float
    total_processing_time_hours: float

class WorkflowROIReport(BaseModel):
    report_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    tenant_id: str
    
    # Report period
    start_date: datetime
    end_date: datetime
    
    # Summary metrics
    total_workflows: int
    total_revenue_impact: float
    total_cost_savings: float
    total_time_saved_hours: float
    overall_roi_percentage: float
    
    # Top performing workflows
    top_revenue_workflows: List[WorkflowROIMetric] = Field(default_factory=list)
    top_efficiency_workflows: List[WorkflowROIMetric] = Field(default_factory=list)
    
    # Category breakdown
    roi_by_category: Dict[str, Dict[str, float]] = Field(default_factory=dict)
    
    generated_at: datetime = Field(default_factory=datetime.utcnow)

# In-memory storage
workflow_roi_metrics_store: Dict[str, WorkflowROIMetric] = {}
workflow_usage_data_store: Dict[str, WorkflowUsageData] = {}
roi_reports_store: Dict[str, WorkflowROIReport] = {}

class WorkflowROICalculator:
    def __init__(self):
        # Default cost assumptions
        self.avg_hourly_rate = 75.0  # Average employee hourly rate
        self.implementation_cost_base = 10000.0  # Base implementation cost
        self.monthly_operational_cost_base = 500.0  # Base monthly operational cost
        
    def calculate_workflow_roi(self, tenant_id: str, workflow_id: str, workflow_name: str,
                             template_id: str, template_name: str, 
                             workflow_category: WorkflowCategory,
                             usage_data: WorkflowUsageData) -> WorkflowROIMetric:
        """Calculate ROI metrics for a specific workflow"""
        
        # Calculate success rate
        success_rate = (usage_data.successful_executions / usage_data.executions * 100) if usage_data.executions > 0 else 0.0
        
        # Calculate time savings (assume 30% time reduction per execution)
        time_saved_per_execution = usage_data.avg_execution_time_minutes * 0.3 / 60  # hours
        total_time_saved = time_saved_per_execution * usage_data.successful_executions
        
        # Calculate cost savings from time savings
        cost_savings = total_time_saved * self.avg_hourly_rate
        
        # Calculate revenue impact based on workflow category
        revenue_impact = self._calculate_revenue_impact(
            workflow_category, usage_data.successful_executions, success_rate
        )
        
        # Calculate investment costs
        implementation_cost = self.implementation_cost_base
        monthly_operational_cost = self.monthly_operational_cost_base
        total_investment = implementation_cost + monthly_operational_cost
        
        # Calculate ROI metrics
        total_benefit = revenue_impact + cost_savings
        net_benefit = total_benefit - total_investment
        roi_percentage = (net_benefit / total_investment * 100) if total_investment > 0 else 0.0
        payback_months = (total_investment / (total_benefit / 12)) if total_benefit > 0 else float('inf')
        
        # Calculate per-execution metrics
        revenue_per_execution = revenue_impact / usage_data.executions if usage_data.executions > 0 else 0.0
        cost_per_execution = total_investment / usage_data.executions if usage_data.executions > 0 else 0.0
        
        # Calculate efficiency gain
        efficiency_percentage = (time_saved_per_execution / (usage_data.avg_execution_time_minutes / 60)) * 100 if usage_data.avg_execution_time_minutes > 0 else 0.0
        
        return WorkflowROIMetric(
            tenant_id=tenant_id,
            workflow_id=workflow_id,
            workflow_name=workflow_name,
            template_id=template_id,
            template_name=template_name,
            workflow_category=workflow_category,
            revenue_impact=revenue_impact,
            cost_savings=cost_savings,
            time_saved_hours=total_time_saved,
            efficiency_percentage=efficiency_percentage,
            implementation_cost=implementation_cost,
            operational_cost_monthly=monthly_operational_cost,
            total_investment=total_investment,
            total_benefit=total_benefit,
            net_benefit=net_benefit,
            roi_percentage=roi_percentage,
            payback_months=payback_months,
            executions_count=usage_data.executions,
            active_users=usage_data.active_users,
            success_rate=success_rate,
            revenue_per_execution=revenue_per_execution,
            cost_per_execution=cost_per_execution,
            time_saved_per_execution=time_saved_per_execution
        )
    
    def _calculate_revenue_impact(self, category: WorkflowCategory, executions: int, success_rate: float) -> float:
        """Calculate revenue impact based on workflow category"""
        
        # Revenue impact multipliers by category
        category_multipliers = {
            WorkflowCategory.SALES: 1500.0,  # $1500 per successful execution
            WorkflowCategory.MARKETING: 800.0,  # $800 per successful execution
            WorkflowCategory.OPERATIONS: 400.0,  # $400 per successful execution
            WorkflowCategory.FINANCE: 600.0,  # $600 per successful execution
            WorkflowCategory.CUSTOMER_SUCCESS: 1000.0  # $1000 per successful execution
        }
        
        multiplier = category_multipliers.get(category, 500.0)
        successful_executions = executions * (success_rate / 100)
        
        return successful_executions * multiplier

# Global calculator instance
calculator = WorkflowROICalculator()

@app.post("/workflow-roi/usage/record", response_model=WorkflowUsageData)
async def record_workflow_usage(usage_data: WorkflowUsageData):
    """Record workflow usage data"""
    
    workflow_usage_data_store[usage_data.workflow_id] = usage_data
    
    logger.info(f"✅ Recorded usage data for workflow {usage_data.workflow_id} - {usage_data.executions} executions")
    return usage_data

@app.post("/workflow-roi/calculate", response_model=WorkflowROIMetric)
async def calculate_workflow_roi(
    tenant_id: str,
    workflow_id: str,
    workflow_name: str,
    template_id: str,
    template_name: str,
    workflow_category: WorkflowCategory
):
    """Calculate ROI metrics for a workflow"""
    
    # Get usage data
    usage_data = workflow_usage_data_store.get(workflow_id)
    if not usage_data:
        raise HTTPException(status_code=404, detail="Usage data not found for workflow")
    
    # Calculate ROI
    roi_metric = calculator.calculate_workflow_roi(
        tenant_id, workflow_id, workflow_name, template_id, template_name,
        workflow_category, usage_data
    )
    
    # Store metric
    workflow_roi_metrics_store[roi_metric.metric_id] = roi_metric
    
    logger.info(f"✅ Calculated ROI for workflow {workflow_name} - ROI: {roi_metric.roi_percentage:.1f}%")
    return roi_metric

@app.get("/workflow-roi/tenant/{tenant_id}")
async def get_tenant_workflow_roi(
    tenant_id: str,
    workflow_category: Optional[WorkflowCategory] = None
):
    """Get workflow ROI metrics for a tenant"""
    
    tenant_metrics = [
        metric for metric in workflow_roi_metrics_store.values()
        if metric.tenant_id == tenant_id
    ]
    
    if workflow_category:
        tenant_metrics = [m for m in tenant_metrics if m.workflow_category == workflow_category]
    
    # Sort by ROI percentage (descending)
    tenant_metrics.sort(key=lambda x: x.roi_percentage, reverse=True)
    
    return {
        "tenant_id": tenant_id,
        "workflow_count": len(tenant_metrics),
        "total_revenue_impact": sum([m.revenue_impact for m in tenant_metrics]),
        "total_cost_savings": sum([m.cost_savings for m in tenant_metrics]),
        "average_roi": sum([m.roi_percentage for m in tenant_metrics]) / len(tenant_metrics) if tenant_metrics else 0.0,
        "workflows": tenant_metrics
    }

@app.get("/workflow-roi/workflow/{workflow_id}")
async def get_workflow_roi(workflow_id: str):
    """Get ROI metrics for a specific workflow"""
    
    workflow_metrics = [
        metric for metric in workflow_roi_metrics_store.values()
        if metric.workflow_id == workflow_id
    ]
    
    if not workflow_metrics:
        raise HTTPException(status_code=404, detail="No ROI metrics found for workflow")
    
    # Return most recent metric
    latest_metric = max(workflow_metrics, key=lambda x: x.measurement_date)
    
    return latest_metric

@app.post("/workflow-roi/report", response_model=WorkflowROIReport)
async def generate_workflow_roi_report(
    tenant_id: str,
    start_date: datetime,
    end_date: datetime
):
    """Generate workflow ROI report"""
    
    # Get metrics for the period
    period_metrics = [
        metric for metric in workflow_roi_metrics_store.values()
        if (metric.tenant_id == tenant_id and 
            start_date <= metric.measurement_date <= end_date)
    ]
    
    if not period_metrics:
        raise HTTPException(status_code=404, detail="No ROI metrics found for the specified period")
    
    # Calculate summary metrics
    total_workflows = len(period_metrics)
    total_revenue_impact = sum([m.revenue_impact for m in period_metrics])
    total_cost_savings = sum([m.cost_savings for m in period_metrics])
    total_time_saved = sum([m.time_saved_hours for m in period_metrics])
    overall_roi = sum([m.roi_percentage for m in period_metrics]) / len(period_metrics)
    
    # Top performing workflows
    top_revenue = sorted(period_metrics, key=lambda x: x.revenue_impact, reverse=True)[:5]
    top_efficiency = sorted(period_metrics, key=lambda x: x.efficiency_percentage, reverse=True)[:5]
    
    # Category breakdown
    roi_by_category = {}
    for category in WorkflowCategory:
        category_metrics = [m for m in period_metrics if m.workflow_category == category]
        if category_metrics:
            roi_by_category[category.value] = {
                "workflows": len(category_metrics),
                "revenue_impact": sum([m.revenue_impact for m in category_metrics]),
                "cost_savings": sum([m.cost_savings for m in category_metrics]),
                "average_roi": sum([m.roi_percentage for m in category_metrics]) / len(category_metrics)
            }
    
    report = WorkflowROIReport(
        tenant_id=tenant_id,
        start_date=start_date,
        end_date=end_date,
        total_workflows=total_workflows,
        total_revenue_impact=total_revenue_impact,
        total_cost_savings=total_cost_savings,
        total_time_saved_hours=total_time_saved,
        overall_roi_percentage=overall_roi,
        top_revenue_workflows=top_revenue,
        top_efficiency_workflows=top_efficiency,
        roi_by_category=roi_by_category
    )
    
    roi_reports_store[report.report_id] = report
    
    logger.info(f"✅ Generated workflow ROI report for tenant {tenant_id} - Overall ROI: {overall_roi:.1f}%")
    return report

@app.get("/workflow-roi/summary")
async def get_workflow_roi_summary():
    """Get overall workflow ROI summary"""
    
    total_metrics = len(workflow_roi_metrics_store)
    
    if total_metrics == 0:
        return {
            "total_workflows": 0,
            "total_tenants": 0,
            "total_revenue_impact": 0.0,
            "average_roi": 0.0
        }
    
    # Calculate totals
    all_metrics = list(workflow_roi_metrics_store.values())
    total_revenue_impact = sum([m.revenue_impact for m in all_metrics])
    total_cost_savings = sum([m.cost_savings for m in all_metrics])
    average_roi = sum([m.roi_percentage for m in all_metrics]) / len(all_metrics)
    
    # Category breakdown
    category_breakdown = {}
    for category in WorkflowCategory:
        category_metrics = [m for m in all_metrics if m.workflow_category == category]
        category_breakdown[category.value] = {
            "workflows": len(category_metrics),
            "average_roi": sum([m.roi_percentage for m in category_metrics]) / len(category_metrics) if category_metrics else 0.0
        }
    
    return {
        "total_workflows": len(set(m.workflow_id for m in all_metrics)),
        "total_tenants": len(set(m.tenant_id for m in all_metrics)),
        "total_revenue_impact": round(total_revenue_impact, 2),
        "total_cost_savings": round(total_cost_savings, 2),
        "average_roi": round(average_roi, 2),
        "category_breakdown": category_breakdown,
        "total_reports": len(roi_reports_store)
    }
