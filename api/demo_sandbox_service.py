"""
Task 3.3.44: Multi-Tenant Demo Sandbox Service
- Isolated sandbox infrastructure with dummy data for safe trials
- No PII risk, safe exploration across all UX modes
"""

from fastapi import FastAPI, HTTPException, BackgroundTasks
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from datetime import datetime, timedelta
from enum import Enum
import uuid
import logging
import random

app = FastAPI(title="RBIA Multi-Tenant Demo Sandbox Service")
logger = logging.getLogger(__name__)

class UXMode(str, Enum):
    UI_LED = "ui_led"
    ASSISTED = "assisted"
    CONVERSATIONAL = "conversational"

class SandboxStatus(str, Enum):
    CREATING = "creating"
    ACTIVE = "active"
    SUSPENDED = "suspended"
    EXPIRED = "expired"
    DELETED = "deleted"

class DatasetType(str, Enum):
    SALES_PIPELINE = "sales_pipeline"
    CUSTOMER_DATA = "customer_data"
    FINANCIAL_METRICS = "financial_metrics"
    MARKETING_CAMPAIGNS = "marketing_campaigns"
    SUPPORT_TICKETS = "support_tickets"
    HR_ANALYTICS = "hr_analytics"

class DemoSandbox(BaseModel):
    sandbox_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    tenant_id: str
    
    # Sandbox configuration
    sandbox_name: str
    description: str
    enabled_modes: List[UXMode] = Field(default_factory=lambda: [UXMode.UI_LED, UXMode.ASSISTED, UXMode.CONVERSATIONAL])
    
    # Data configuration
    included_datasets: List[DatasetType] = Field(default_factory=list)
    data_volume: str = "medium"  # small, medium, large
    
    # Access control
    allowed_users: List[str] = Field(default_factory=list)  # User IDs
    max_concurrent_users: int = 10
    
    # Lifecycle
    status: SandboxStatus = SandboxStatus.CREATING
    created_at: datetime = Field(default_factory=datetime.utcnow)
    expires_at: datetime = Field(default_factory=lambda: datetime.utcnow() + timedelta(days=30))
    
    # Usage tracking
    total_sessions: int = 0
    total_workflows_created: int = 0
    last_activity: Optional[datetime] = None
    
    # Isolation
    isolated_database_url: Optional[str] = None
    isolated_storage_bucket: Optional[str] = None

class SandboxSession(BaseModel):
    session_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    sandbox_id: str
    user_id: str
    
    # Session details
    ux_mode: UXMode
    started_at: datetime = Field(default_factory=datetime.utcnow)
    last_activity: datetime = Field(default_factory=datetime.utcnow)
    
    # Activity tracking
    workflows_created: int = 0
    actions_performed: List[str] = Field(default_factory=list)
    
    # Status
    is_active: bool = True

class DummyDataset(BaseModel):
    dataset_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    sandbox_id: str
    
    # Dataset details
    dataset_type: DatasetType
    dataset_name: str
    record_count: int
    
    # Schema
    columns: List[Dict[str, str]] = Field(default_factory=list)  # name, type, description
    sample_data: List[Dict[str, Any]] = Field(default_factory=list)
    
    # Metadata
    generated_at: datetime = Field(default_factory=datetime.utcnow)
    last_updated: datetime = Field(default_factory=datetime.utcnow)

# In-memory stores (replace with database in production)
demo_sandboxes_store: Dict[str, DemoSandbox] = {}
sandbox_sessions_store: Dict[str, SandboxSession] = {}
dummy_datasets_store: Dict[str, DummyDataset] = {}

def _generate_dummy_sales_data(record_count: int) -> List[Dict[str, Any]]:
    """Generate dummy sales pipeline data"""
    
    companies = ["Acme Corp", "TechFlow Inc", "Global Solutions", "Innovation Labs", "DataCorp", "CloudTech", "NextGen Systems"]
    stages = ["Prospecting", "Qualification", "Proposal", "Negotiation", "Closed Won", "Closed Lost"]
    industries = ["Technology", "Healthcare", "Finance", "Manufacturing", "Retail", "Education"]
    
    dummy_data = []
    for i in range(record_count):
        record = {
            "deal_id": f"DEAL-{1000 + i}",
            "company_name": random.choice(companies),
            "deal_value": random.randint(5000, 500000),
            "stage": random.choice(stages),
            "probability": random.randint(10, 95),
            "industry": random.choice(industries),
            "created_date": (datetime.utcnow() - timedelta(days=random.randint(1, 365))).isoformat(),
            "expected_close_date": (datetime.utcnow() + timedelta(days=random.randint(1, 180))).isoformat(),
            "owner": f"sales_rep_{random.randint(1, 10)}",
            "last_activity": (datetime.utcnow() - timedelta(days=random.randint(1, 30))).isoformat()
        }
        dummy_data.append(record)
    
    return dummy_data

def _generate_dummy_customer_data(record_count: int) -> List[Dict[str, Any]]:
    """Generate dummy customer data (anonymized)"""
    
    segments = ["Enterprise", "SMB", "Startup", "Government"]
    statuses = ["Active", "Churned", "At Risk", "Growing"]
    
    dummy_data = []
    for i in range(record_count):
        record = {
            "customer_id": f"CUST-{2000 + i}",
            "company_name": f"Customer Company {i+1}",  # Anonymized
            "segment": random.choice(segments),
            "status": random.choice(statuses),
            "monthly_recurring_revenue": random.randint(1000, 50000),
            "contract_start_date": (datetime.utcnow() - timedelta(days=random.randint(30, 1095))).isoformat(),
            "last_login": (datetime.utcnow() - timedelta(days=random.randint(1, 60))).isoformat(),
            "support_tickets_count": random.randint(0, 20),
            "nps_score": random.randint(1, 10),
            "region": random.choice(["North America", "Europe", "Asia Pacific"])
        }
        dummy_data.append(record)
    
    return dummy_data

def _generate_dummy_financial_data(record_count: int) -> List[Dict[str, Any]]:
    """Generate dummy financial metrics"""
    
    metrics = ["Revenue", "Expenses", "Profit", "Cash Flow"]
    departments = ["Sales", "Marketing", "Engineering", "Operations", "Support"]
    
    dummy_data = []
    for i in range(record_count):
        record = {
            "metric_id": f"FIN-{3000 + i}",
            "metric_type": random.choice(metrics),
            "department": random.choice(departments),
            "amount": random.randint(10000, 1000000),
            "period": f"2024-{random.randint(1, 12):02d}",
            "budget_variance": random.uniform(-0.2, 0.3),
            "year_over_year_growth": random.uniform(-0.1, 0.4),
            "forecast_accuracy": random.uniform(0.8, 0.98)
        }
        dummy_data.append(record)
    
    return dummy_data

def _create_dummy_dataset(sandbox_id: str, dataset_type: DatasetType, data_volume: str) -> DummyDataset:
    """Create a dummy dataset for the sandbox"""
    
    # Determine record count based on volume
    volume_mapping = {"small": 100, "medium": 500, "large": 2000}
    record_count = volume_mapping.get(data_volume, 500)
    
    if dataset_type == DatasetType.SALES_PIPELINE:
        sample_data = _generate_dummy_sales_data(record_count)
        columns = [
            {"name": "deal_id", "type": "string", "description": "Unique deal identifier"},
            {"name": "company_name", "type": "string", "description": "Company name"},
            {"name": "deal_value", "type": "number", "description": "Deal value in USD"},
            {"name": "stage", "type": "string", "description": "Sales stage"},
            {"name": "probability", "type": "number", "description": "Win probability percentage"}
        ]
        dataset_name = "Demo Sales Pipeline Data"
    
    elif dataset_type == DatasetType.CUSTOMER_DATA:
        sample_data = _generate_dummy_customer_data(record_count)
        columns = [
            {"name": "customer_id", "type": "string", "description": "Unique customer identifier"},
            {"name": "company_name", "type": "string", "description": "Anonymized company name"},
            {"name": "segment", "type": "string", "description": "Customer segment"},
            {"name": "status", "type": "string", "description": "Customer status"},
            {"name": "monthly_recurring_revenue", "type": "number", "description": "MRR in USD"}
        ]
        dataset_name = "Demo Customer Data"
    
    elif dataset_type == DatasetType.FINANCIAL_METRICS:
        sample_data = _generate_dummy_financial_data(record_count)
        columns = [
            {"name": "metric_id", "type": "string", "description": "Unique metric identifier"},
            {"name": "metric_type", "type": "string", "description": "Type of financial metric"},
            {"name": "amount", "type": "number", "description": "Metric amount in USD"},
            {"name": "department", "type": "string", "description": "Department"},
            {"name": "period", "type": "string", "description": "Time period"}
        ]
        dataset_name = "Demo Financial Metrics"
    
    else:
        # Default dataset
        sample_data = [{"id": i, "value": f"demo_value_{i}"} for i in range(record_count)]
        columns = [
            {"name": "id", "type": "number", "description": "Record ID"},
            {"name": "value", "type": "string", "description": "Demo value"}
        ]
        dataset_name = f"Demo {dataset_type.value.replace('_', ' ').title()} Data"
    
    dataset = DummyDataset(
        sandbox_id=sandbox_id,
        dataset_type=dataset_type,
        dataset_name=dataset_name,
        record_count=len(sample_data),
        columns=columns,
        sample_data=sample_data[:10]  # Store only sample for API responses
    )
    
    return dataset

@app.post("/sandboxes", response_model=DemoSandbox)
async def create_demo_sandbox(
    tenant_id: str,
    sandbox_name: str,
    description: str,
    included_datasets: List[DatasetType],
    data_volume: str = "medium",
    duration_days: int = 30,
    background_tasks: BackgroundTasks = BackgroundTasks()
):
    """Create a new demo sandbox environment"""
    
    sandbox = DemoSandbox(
        tenant_id=tenant_id,
        sandbox_name=sandbox_name,
        description=description,
        included_datasets=included_datasets,
        data_volume=data_volume,
        expires_at=datetime.utcnow() + timedelta(days=duration_days),
        isolated_database_url=f"sandbox_db_{uuid.uuid4().hex[:8]}",
        isolated_storage_bucket=f"sandbox-storage-{uuid.uuid4().hex[:8]}"
    )
    
    demo_sandboxes_store[sandbox.sandbox_id] = sandbox
    
    # Generate dummy datasets in background
    background_tasks.add_task(_setup_sandbox_data, sandbox)
    
    logger.info(f"Created demo sandbox: {sandbox_name} for tenant {tenant_id}")
    return sandbox

async def _setup_sandbox_data(sandbox: DemoSandbox):
    """Setup dummy data for the sandbox"""
    
    try:
        # Generate datasets
        for dataset_type in sandbox.included_datasets:
            dataset = _create_dummy_dataset(sandbox.sandbox_id, dataset_type, sandbox.data_volume)
            dummy_datasets_store[dataset.dataset_id] = dataset
        
        # Mark sandbox as active
        sandbox.status = SandboxStatus.ACTIVE
        logger.info(f"Sandbox {sandbox.sandbox_id} setup completed with {len(sandbox.included_datasets)} datasets")
        
    except Exception as e:
        sandbox.status = SandboxStatus.SUSPENDED
        logger.error(f"Failed to setup sandbox {sandbox.sandbox_id}: {e}")

@app.get("/sandboxes", response_model=List[DemoSandbox])
async def get_demo_sandboxes(
    tenant_id: str,
    status: Optional[SandboxStatus] = None
):
    """Get demo sandboxes for a tenant"""
    
    sandboxes = [s for s in demo_sandboxes_store.values() if s.tenant_id == tenant_id]
    
    if status:
        sandboxes = [s for s in sandboxes if s.status == status]
    
    return sandboxes

@app.post("/sandboxes/{sandbox_id}/sessions", response_model=SandboxSession)
async def start_sandbox_session(
    sandbox_id: str,
    user_id: str,
    ux_mode: UXMode
):
    """Start a new session in the demo sandbox"""
    
    if sandbox_id not in demo_sandboxes_store:
        raise HTTPException(status_code=404, detail="Sandbox not found")
    
    sandbox = demo_sandboxes_store[sandbox_id]
    
    if sandbox.status != SandboxStatus.ACTIVE:
        raise HTTPException(status_code=400, detail="Sandbox is not active")
    
    if datetime.utcnow() > sandbox.expires_at:
        raise HTTPException(status_code=400, detail="Sandbox has expired")
    
    # Check concurrent user limit
    active_sessions = [
        s for s in sandbox_sessions_store.values()
        if s.sandbox_id == sandbox_id and s.is_active
    ]
    
    if len(active_sessions) >= sandbox.max_concurrent_users:
        raise HTTPException(status_code=429, detail="Maximum concurrent users reached")
    
    # Create session
    session = SandboxSession(
        sandbox_id=sandbox_id,
        user_id=user_id,
        ux_mode=ux_mode
    )
    
    sandbox_sessions_store[session.session_id] = session
    
    # Update sandbox stats
    sandbox.total_sessions += 1
    sandbox.last_activity = datetime.utcnow()
    
    logger.info(f"Started sandbox session {session.session_id} for user {user_id} in {ux_mode.value} mode")
    return session

@app.get("/sandboxes/{sandbox_id}/datasets")
async def get_sandbox_datasets(sandbox_id: str):
    """Get available datasets in the sandbox"""
    
    if sandbox_id not in demo_sandboxes_store:
        raise HTTPException(status_code=404, detail="Sandbox not found")
    
    datasets = [
        dataset for dataset in dummy_datasets_store.values()
        if dataset.sandbox_id == sandbox_id
    ]
    
    return {"sandbox_id": sandbox_id, "datasets": datasets}

@app.get("/sandboxes/{sandbox_id}/datasets/{dataset_type}/sample")
async def get_dataset_sample(sandbox_id: str, dataset_type: DatasetType):
    """Get sample data from a specific dataset"""
    
    dataset = None
    for ds in dummy_datasets_store.values():
        if ds.sandbox_id == sandbox_id and ds.dataset_type == dataset_type:
            dataset = ds
            break
    
    if not dataset:
        raise HTTPException(status_code=404, detail="Dataset not found")
    
    return {
        "dataset_name": dataset.dataset_name,
        "dataset_type": dataset.dataset_type,
        "record_count": dataset.record_count,
        "columns": dataset.columns,
        "sample_data": dataset.sample_data,
        "note": "This is anonymized demo data with no PII"
    }

@app.post("/sandboxes/{sandbox_id}/sessions/{session_id}/activity")
async def log_session_activity(
    sandbox_id: str,
    session_id: str,
    activity_type: str,
    activity_details: Optional[Dict[str, Any]] = None
):
    """Log activity in a sandbox session"""
    
    if session_id not in sandbox_sessions_store:
        raise HTTPException(status_code=404, detail="Session not found")
    
    session = sandbox_sessions_store[session_id]
    session.last_activity = datetime.utcnow()
    session.actions_performed.append(f"{activity_type}: {activity_details or {}}")
    
    # Update sandbox activity
    sandbox = demo_sandboxes_store[sandbox_id]
    sandbox.last_activity = datetime.utcnow()
    
    if activity_type == "workflow_created":
        session.workflows_created += 1
        sandbox.total_workflows_created += 1
    
    return {"status": "logged", "activity_type": activity_type}

@app.post("/sandboxes/{sandbox_id}/sessions/{session_id}/end")
async def end_sandbox_session(sandbox_id: str, session_id: str):
    """End a sandbox session"""
    
    if session_id not in sandbox_sessions_store:
        raise HTTPException(status_code=404, detail="Session not found")
    
    session = sandbox_sessions_store[session_id]
    session.is_active = False
    
    session_duration = (datetime.utcnow() - session.started_at).total_seconds()
    
    logger.info(f"Ended sandbox session {session_id}, duration: {session_duration}s, workflows created: {session.workflows_created}")
    
    return {
        "status": "ended",
        "session_duration_seconds": session_duration,
        "workflows_created": session.workflows_created,
        "actions_performed": len(session.actions_performed)
    }

@app.get("/sandboxes/{sandbox_id}/analytics")
async def get_sandbox_analytics(sandbox_id: str):
    """Get usage analytics for a sandbox"""
    
    if sandbox_id not in demo_sandboxes_store:
        raise HTTPException(status_code=404, detail="Sandbox not found")
    
    sandbox = demo_sandboxes_store[sandbox_id]
    
    # Get all sessions for this sandbox
    sessions = [
        session for session in sandbox_sessions_store.values()
        if session.sandbox_id == sandbox_id
    ]
    
    active_sessions = [s for s in sessions if s.is_active]
    
    # Calculate analytics
    total_session_time = sum(
        (s.last_activity - s.started_at).total_seconds() 
        for s in sessions if not s.is_active
    )
    
    analytics = {
        "sandbox_id": sandbox_id,
        "sandbox_name": sandbox.sandbox_name,
        "status": sandbox.status.value,
        "created_at": sandbox.created_at,
        "expires_at": sandbox.expires_at,
        "total_sessions": len(sessions),
        "active_sessions": len(active_sessions),
        "total_workflows_created": sandbox.total_workflows_created,
        "total_session_time_seconds": total_session_time,
        "avg_session_duration": total_session_time / len(sessions) if sessions else 0,
        "usage_by_mode": {},
        "datasets_available": len([d for d in dummy_datasets_store.values() if d.sandbox_id == sandbox_id])
    }
    
    # Usage by UX mode
    for mode in UXMode:
        mode_sessions = [s for s in sessions if s.ux_mode == mode]
        analytics["usage_by_mode"][mode.value] = len(mode_sessions)
    
    return analytics

@app.delete("/sandboxes/{sandbox_id}")
async def delete_sandbox(sandbox_id: str):
    """Delete a demo sandbox and all its data"""
    
    if sandbox_id not in demo_sandboxes_store:
        raise HTTPException(status_code=404, detail="Sandbox not found")
    
    sandbox = demo_sandboxes_store[sandbox_id]
    sandbox.status = SandboxStatus.DELETED
    
    # End all active sessions
    for session in sandbox_sessions_store.values():
        if session.sandbox_id == sandbox_id and session.is_active:
            session.is_active = False
    
    # Delete datasets
    datasets_to_delete = [
        dataset_id for dataset_id, dataset in dummy_datasets_store.items()
        if dataset.sandbox_id == sandbox_id
    ]
    
    for dataset_id in datasets_to_delete:
        del dummy_datasets_store[dataset_id]
    
    logger.info(f"Deleted sandbox {sandbox_id} and {len(datasets_to_delete)} datasets")
    
    return {"status": "deleted", "sandbox_id": sandbox_id, "datasets_deleted": len(datasets_to_delete)}

@app.get("/analytics/sandbox-usage")
async def get_overall_sandbox_analytics(tenant_id: str):
    """Get overall sandbox usage analytics for tenant"""
    
    tenant_sandboxes = [s for s in demo_sandboxes_store.values() if s.tenant_id == tenant_id]
    
    if not tenant_sandboxes:
        return {"message": "No sandboxes found for tenant"}
    
    analytics = {
        "total_sandboxes": len(tenant_sandboxes),
        "active_sandboxes": len([s for s in tenant_sandboxes if s.status == SandboxStatus.ACTIVE]),
        "total_sessions": sum(s.total_sessions for s in tenant_sandboxes),
        "total_workflows_created": sum(s.total_workflows_created for s in tenant_sandboxes),
        "by_status": {},
        "by_dataset_type": {},
        "avg_sandbox_lifetime_days": 0
    }
    
    # Analyze by status
    for status in SandboxStatus:
        status_count = len([s for s in tenant_sandboxes if s.status == status])
        analytics["by_status"][status.value] = status_count
    
    # Analyze by dataset type
    for dataset_type in DatasetType:
        type_count = len([s for s in tenant_sandboxes if dataset_type in s.included_datasets])
        analytics["by_dataset_type"][dataset_type.value] = type_count
    
    # Calculate average sandbox lifetime
    completed_sandboxes = [s for s in tenant_sandboxes if s.status in [SandboxStatus.EXPIRED, SandboxStatus.DELETED]]
    if completed_sandboxes:
        total_lifetime = sum(
            (s.expires_at - s.created_at).days for s in completed_sandboxes
        )
        analytics["avg_sandbox_lifetime_days"] = total_lifetime / len(completed_sandboxes)
    
    return analytics

@app.get("/health")
async def health_check():
    return {"status": "healthy", "service": "RBIA Multi-Tenant Demo Sandbox Service", "task": "3.3.44"}
