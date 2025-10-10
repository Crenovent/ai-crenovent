"""
Automated Adoption Snapshot Email Service - Task 6.1.66
=======================================================
Weekly email reports for CRO/CFO with adoption metrics
"""

from fastapi import FastAPI, BackgroundTasks
from pydantic import BaseModel, Field, EmailStr
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
from enum import Enum
import asyncio

app = FastAPI(title="Adoption Email Service")


class ReportFrequency(str, Enum):
    DAILY = "daily"
    WEEKLY = "weekly"
    MONTHLY = "monthly"


class EmailSubscription(BaseModel):
    """Email subscription for adoption reports"""
    subscription_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    tenant_id: str
    recipient_email: EmailStr
    recipient_name: str
    recipient_role: str  # CRO, CFO, Compliance
    
    # Preferences
    frequency: ReportFrequency = ReportFrequency.WEEKLY
    report_types: List[str] = Field(default_factory=lambda: ['adoption', 'roi', 'trust'])
    
    # Status
    active: bool = True
    created_at: datetime = Field(default_factory=datetime.utcnow)
    last_sent_at: Optional[datetime] = None


class AdoptionSnapshot(BaseModel):
    """Adoption metrics snapshot"""
    tenant_id: str
    period_start: datetime
    period_end: datetime
    
    # Adoption metrics
    total_workflows: int = 0
    active_workflows: int = 0
    total_executions: int = 0
    
    # Performance metrics
    avg_execution_time_ms: float = 0
    success_rate: float = 0
    
    # Business impact
    variance_reduction_pct: float = 0
    cost_savings_usd: float = 0
    
    # Trust metrics
    avg_trust_score: float = 0
    policy_compliance_rate: float = 0


subscriptions: Dict[str, EmailSubscription] = {}
import uuid


def generate_email_html(snapshot: AdoptionSnapshot, recipient: EmailSubscription) -> str:
    """Generate HTML email from snapshot"""
    
    return f"""
<!DOCTYPE html>
<html>
<head>
    <style>
        body {{ font-family: Arial, sans-serif; line-height: 1.6; color: #333; }}
        .header {{ background: #2563eb; color: white; padding: 20px; }}
        .metric-card {{ background: #f3f4f6; padding: 15px; margin: 10px 0; border-radius: 5px; }}
        .metric-value {{ font-size: 24px; font-weight: bold; color: #2563eb; }}
        .metric-label {{ font-size: 14px; color: #6b7280; }}
    </style>
</head>
<body>
    <div class="header">
        <h1>RBIA Adoption Snapshot</h1>
        <p>{snapshot.period_start.strftime('%B %d')} - {snapshot.period_end.strftime('%B %d, %Y')}</p>
    </div>
    
    <div style="padding: 20px;">
        <h2>Hello {recipient.recipient_name},</h2>
        <p>Here's your weekly RBIA adoption snapshot for tenant {snapshot.tenant_id}.</p>
        
        <h3>📊 Adoption Metrics</h3>
        <div class="metric-card">
            <div class="metric-label">Active Workflows</div>
            <div class="metric-value">{snapshot.active_workflows} / {snapshot.total_workflows}</div>
        </div>
        
        <div class="metric-card">
            <div class="metric-label">Total Executions</div>
            <div class="metric-value">{snapshot.total_executions:,}</div>
        </div>
        
        <div class="metric-card">
            <div class="metric-label">Success Rate</div>
            <div class="metric-value">{snapshot.success_rate:.1f}%</div>
        </div>
        
        <h3>💰 Business Impact</h3>
        <div class="metric-card">
            <div class="metric-label">Variance Reduction</div>
            <div class="metric-value">{snapshot.variance_reduction_pct:.1f}%</div>
        </div>
        
        <div class="metric-card">
            <div class="metric-label">Cost Savings</div>
            <div class="metric-value">${snapshot.cost_savings_usd:,.0f}</div>
        </div>
        
        <h3>🛡️ Trust & Governance</h3>
        <div class="metric-card">
            <div class="metric-label">Average Trust Score</div>
            <div class="metric-value">{snapshot.avg_trust_score:.2f}</div>
        </div>
        
        <div class="metric-card">
            <div class="metric-label">Policy Compliance</div>
            <div class="metric-value">{snapshot.policy_compliance_rate:.1f}%</div>
        </div>
        
        <p style="margin-top: 30px;">
            <a href="https://rbia-dashboard.com" style="background: #2563eb; color: white; padding: 10px 20px; text-decoration: none; border-radius: 5px;">
                View Full Dashboard
            </a>
        </p>
        
        <p style="color: #6b7280; font-size: 12px; margin-top: 40px;">
            This is an automated report from RBIA Platform. 
            <a href="#">Unsubscribe</a>
        </p>
    </div>
</body>
</html>
"""


async def send_email(to: str, subject: str, html_content: str):
    """Send email (simulated)"""
    # In production, use SendGrid, AWS SES, or similar
    print(f"📧 Sending email to: {to}")
    print(f"   Subject: {subject}")
    print(f"   Content length: {len(html_content)} chars")
    
    # Simulate email sending delay
    await asyncio.sleep(0.1)
    
    return True


@app.post("/email-subscriptions", response_model=EmailSubscription)
async def create_subscription(subscription: EmailSubscription):
    """Create email subscription for adoption reports"""
    subscriptions[subscription.subscription_id] = subscription
    return subscription


@app.post("/email-subscriptions/{subscription_id}/send-now")
async def send_snapshot_now(subscription_id: str, background_tasks: BackgroundTasks):
    """Manually trigger snapshot email"""
    
    if subscription_id not in subscriptions:
        raise HTTPException(status_code=404, detail="Subscription not found")
    
    subscription = subscriptions[subscription_id]
    
    # Generate snapshot
    snapshot = AdoptionSnapshot(
        tenant_id=subscription.tenant_id,
        period_start=datetime.utcnow() - timedelta(days=7),
        period_end=datetime.utcnow(),
        total_workflows=25,
        active_workflows=18,
        total_executions=1250,
        avg_execution_time_ms=850.0,
        success_rate=96.5,
        variance_reduction_pct=23.5,
        cost_savings_usd=45000,
        avg_trust_score=0.87,
        policy_compliance_rate=98.2
    )
    
    # Generate email
    html_content = generate_email_html(snapshot, subscription)
    
    # Send email in background
    background_tasks.add_task(
        send_email,
        subscription.recipient_email,
        f"RBIA Adoption Snapshot - {datetime.utcnow().strftime('%B %d, %Y')}",
        html_content
    )
    
    # Update last sent
    subscription.last_sent_at = datetime.utcnow()
    
    return {
        'status': 'success',
        'message': 'Email queued for delivery',
        'recipient': subscription.recipient_email
    }


@app.post("/email-subscriptions/send-batch")
async def send_batch_emails(background_tasks: BackgroundTasks):
    """Send emails to all active subscriptions (scheduled job)"""
    
    sent_count = 0
    
    for subscription in subscriptions.values():
        if not subscription.active:
            continue
        
        # Check if due to send
        if subscription.last_sent_at:
            days_since_last = (datetime.utcnow() - subscription.last_sent_at).days
            
            if subscription.frequency == ReportFrequency.DAILY and days_since_last < 1:
                continue
            elif subscription.frequency == ReportFrequency.WEEKLY and days_since_last < 7:
                continue
            elif subscription.frequency == ReportFrequency.MONTHLY and days_since_last < 30:
                continue
        
        # Send email
        await send_snapshot_now(subscription.subscription_id, background_tasks)
        sent_count += 1
    
    return {
        'status': 'success',
        'emails_sent': sent_count
    }

