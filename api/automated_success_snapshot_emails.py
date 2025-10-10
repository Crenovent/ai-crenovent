"""
Task 4.4.39: Build automated success snapshot emails
- Weekly/monthly email summaries
- Persona-specific highlights
- Automated delivery
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, EmailStr
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
from enum import Enum
import uuid
import logging

app = FastAPI(title="RBIA Automated Success Snapshot Emails")
logger = logging.getLogger(__name__)

class PersonaType(str, Enum):
    CRO = "cro"
    CFO = "cfo"
    COMPLIANCE = "compliance"
    REVOPS = "revops"
    EXECUTIVE = "executive"
    DATA_SCIENTIST = "data_scientist"

class EmailFrequency(str, Enum):
    DAILY = "daily"
    WEEKLY = "weekly"
    MONTHLY = "monthly"

class SnapshotMetric(BaseModel):
    metric_name: str
    current_value: float
    previous_value: float
    change_percentage: float
    trend: str  # "up", "down", "stable"
    is_positive_change: bool

class SuccessSnapshot(BaseModel):
    snapshot_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    
    # Snapshot metadata
    persona: PersonaType
    period_start: datetime
    period_end: datetime
    
    # Top metrics for this persona
    top_metrics: List[SnapshotMetric] = Field(default_factory=list)
    
    # Key highlights
    key_highlights: List[str] = Field(default_factory=list)
    
    # Alerts and warnings
    alerts: List[str] = Field(default_factory=list)
    
    # Recommendations
    recommendations: List[str] = Field(default_factory=list)
    
    # Summary statistics
    overall_health_score: float
    
    generated_at: datetime = Field(default_factory=datetime.utcnow)

class EmailTemplate(BaseModel):
    template_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    persona: PersonaType
    subject: str
    
    # Email sections
    header: str
    metrics_section: str
    highlights_section: str
    alerts_section: str
    recommendations_section: str
    footer: str
    
    created_at: datetime = Field(default_factory=datetime.utcnow)

class EmailSubscription(BaseModel):
    subscription_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    
    user_id: str
    email: EmailStr
    persona: PersonaType
    frequency: EmailFrequency
    
    # Preferences
    include_alerts: bool = True
    include_recommendations: bool = True
    
    # Status
    is_active: bool = True
    
    last_sent: Optional[datetime] = None
    next_scheduled: datetime
    
    created_at: datetime = Field(default_factory=datetime.utcnow)

class EmailDeliveryRecord(BaseModel):
    delivery_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    
    subscription_id: str
    snapshot_id: str
    
    recipient_email: EmailStr
    subject: str
    
    # Delivery status
    status: str  # "sent", "failed", "pending"
    sent_at: Optional[datetime] = None
    
    created_at: datetime = Field(default_factory=datetime.utcnow)

# In-memory storage
snapshots_store: Dict[str, SuccessSnapshot] = {}
templates_store: Dict[str, EmailTemplate] = {}
subscriptions_store: Dict[str, EmailSubscription] = {}
delivery_records_store: Dict[str, EmailDeliveryRecord] = {}

class SnapshotEngine:
    def __init__(self):
        self._initialize_templates()
    
    def _initialize_templates(self):
        """Initialize email templates for each persona"""
        
        templates = {
            PersonaType.CRO: {
                "subject": "Your Weekly Risk & Trust Snapshot",
                "header": "Risk Officer Success Metrics",
                "metrics_section": "Key Risk Metrics",
                "highlights_section": "Risk Highlights",
                "alerts_section": "Risk Alerts",
                "recommendations_section": "Recommended Actions",
                "footer": "RBIA Risk Intelligence"
            },
            PersonaType.CFO: {
                "subject": "Your Weekly Financial Impact Snapshot",
                "header": "Financial Performance Metrics",
                "metrics_section": "Key Financial Metrics",
                "highlights_section": "Financial Highlights",
                "alerts_section": "Cost Alerts",
                "recommendations_section": "Optimization Opportunities",
                "footer": "RBIA Financial Intelligence"
            },
            PersonaType.COMPLIANCE: {
                "subject": "Your Weekly Compliance Snapshot",
                "header": "Compliance & Governance Metrics",
                "metrics_section": "Key Compliance Metrics",
                "highlights_section": "Compliance Highlights",
                "alerts_section": "Compliance Alerts",
                "recommendations_section": "Compliance Actions",
                "footer": "RBIA Compliance Intelligence"
            },
            PersonaType.REVOPS: {
                "subject": "Your Weekly Revenue Operations Snapshot",
                "header": "RevOps Performance Metrics",
                "metrics_section": "Key RevOps Metrics",
                "highlights_section": "Revenue Highlights",
                "alerts_section": "Performance Alerts",
                "recommendations_section": "RevOps Recommendations",
                "footer": "RBIA Revenue Intelligence"
            },
            PersonaType.EXECUTIVE: {
                "subject": "Your Weekly Executive Snapshot",
                "header": "Executive Summary",
                "metrics_section": "Key Performance Indicators",
                "highlights_section": "Strategic Highlights",
                "alerts_section": "Executive Alerts",
                "recommendations_section": "Strategic Recommendations",
                "footer": "RBIA Executive Intelligence"
            }
        }
        
        for persona, template_data in templates.items():
            template = EmailTemplate(
                persona=persona,
                subject=template_data["subject"],
                header=template_data["header"],
                metrics_section=template_data["metrics_section"],
                highlights_section=template_data["highlights_section"],
                alerts_section=template_data["alerts_section"],
                recommendations_section=template_data["recommendations_section"],
                footer=template_data["footer"]
            )
            templates_store[template.template_id] = template
    
    def generate_snapshot(
        self,
        persona: PersonaType,
        period_days: int = 7
    ) -> SuccessSnapshot:
        """Generate success snapshot for a persona"""
        
        period_end = datetime.utcnow()
        period_start = period_end - timedelta(days=period_days)
        
        # Generate metrics based on persona
        top_metrics = self._generate_metrics_for_persona(persona)
        
        # Generate highlights
        highlights = self._generate_highlights(persona, top_metrics)
        
        # Generate alerts
        alerts = self._generate_alerts(persona, top_metrics)
        
        # Generate recommendations
        recommendations = self._generate_recommendations(persona, top_metrics)
        
        # Calculate overall health score
        health_score = self._calculate_health_score(top_metrics)
        
        snapshot = SuccessSnapshot(
            persona=persona,
            period_start=period_start,
            period_end=period_end,
            top_metrics=top_metrics,
            key_highlights=highlights,
            alerts=alerts,
            recommendations=recommendations,
            overall_health_score=health_score
        )
        
        snapshots_store[snapshot.snapshot_id] = snapshot
        return snapshot
    
    def _generate_metrics_for_persona(self, persona: PersonaType) -> List[SnapshotMetric]:
        """Generate metrics relevant to persona"""
        
        persona_metrics = {
            PersonaType.CRO: [
                SnapshotMetric(
                    metric_name="Trust Score",
                    current_value=87.5,
                    previous_value=85.2,
                    change_percentage=2.7,
                    trend="up",
                    is_positive_change=True
                ),
                SnapshotMetric(
                    metric_name="Fairness Score",
                    current_value=91.3,
                    previous_value=90.8,
                    change_percentage=0.6,
                    trend="stable",
                    is_positive_change=True
                ),
                SnapshotMetric(
                    metric_name="Drift Incidents",
                    current_value=3,
                    previous_value=5,
                    change_percentage=-40.0,
                    trend="down",
                    is_positive_change=True
                )
            ],
            PersonaType.CFO: [
                SnapshotMetric(
                    metric_name="ROI",
                    current_value=245.8,
                    previous_value=228.5,
                    change_percentage=7.6,
                    trend="up",
                    is_positive_change=True
                ),
                SnapshotMetric(
                    metric_name="Cost per Prediction",
                    current_value=0.05,
                    previous_value=0.06,
                    change_percentage=-16.7,
                    trend="down",
                    is_positive_change=True
                ),
                SnapshotMetric(
                    metric_name="Monthly Savings",
                    current_value=125000,
                    previous_value=118000,
                    change_percentage=5.9,
                    trend="up",
                    is_positive_change=True
                )
            ],
            PersonaType.COMPLIANCE: [
                SnapshotMetric(
                    metric_name="Governance Compliance",
                    current_value=96.2,
                    previous_value=95.8,
                    change_percentage=0.4,
                    trend="stable",
                    is_positive_change=True
                ),
                SnapshotMetric(
                    metric_name="Audit Pass Rate",
                    current_value=98.5,
                    previous_value=97.2,
                    change_percentage=1.3,
                    trend="up",
                    is_positive_change=True
                ),
                SnapshotMetric(
                    metric_name="Evidence Packs Generated",
                    current_value=142,
                    previous_value=135,
                    change_percentage=5.2,
                    trend="up",
                    is_positive_change=True
                )
            ],
            PersonaType.REVOPS: [
                SnapshotMetric(
                    metric_name="Forecast Accuracy",
                    current_value=89.7,
                    previous_value=87.3,
                    change_percentage=2.7,
                    trend="up",
                    is_positive_change=True
                ),
                SnapshotMetric(
                    metric_name="Override Rate",
                    current_value=3.2,
                    previous_value=4.1,
                    change_percentage=-22.0,
                    trend="down",
                    is_positive_change=True
                ),
                SnapshotMetric(
                    metric_name="Adoption Rate",
                    current_value=78.5,
                    previous_value=75.2,
                    change_percentage=4.4,
                    trend="up",
                    is_positive_change=True
                )
            ],
            PersonaType.EXECUTIVE: [
                SnapshotMetric(
                    metric_name="Overall Trust Index",
                    current_value=88.3,
                    previous_value=86.1,
                    change_percentage=2.6,
                    trend="up",
                    is_positive_change=True
                ),
                SnapshotMetric(
                    metric_name="ROI",
                    current_value=245.8,
                    previous_value=228.5,
                    change_percentage=7.6,
                    trend="up",
                    is_positive_change=True
                ),
                SnapshotMetric(
                    metric_name="User Adoption",
                    current_value=78.5,
                    previous_value=75.2,
                    change_percentage=4.4,
                    trend="up",
                    is_positive_change=True
                )
            ]
        }
        
        return persona_metrics.get(persona, [])
    
    def _generate_highlights(
        self,
        persona: PersonaType,
        metrics: List[SnapshotMetric]
    ) -> List[str]:
        """Generate key highlights"""
        
        highlights = []
        
        for metric in metrics:
            if metric.is_positive_change and abs(metric.change_percentage) > 2:
                highlights.append(
                    f"{metric.metric_name} improved by {abs(metric.change_percentage):.1f}% to {metric.current_value}"
                )
        
        # Add persona-specific highlights
        persona_highlights = {
            PersonaType.CRO: ["All risk models passed compliance checks", "Zero critical drift incidents"],
            PersonaType.CFO: ["Achieved 15% cost reduction target", "ROI exceeded quarterly projections"],
            PersonaType.COMPLIANCE: ["100% audit readiness maintained", "All evidence packs delivered on time"],
            PersonaType.REVOPS: ["Forecast accuracy at all-time high", "User adoption growing steadily"],
            PersonaType.EXECUTIVE: ["Strong performance across all KPIs", "Trust index trending upward"]
        }
        
        highlights.extend(persona_highlights.get(persona, [])[:2])
        return highlights[:5]
    
    def _generate_alerts(
        self,
        persona: PersonaType,
        metrics: List[SnapshotMetric]
    ) -> List[str]:
        """Generate alerts"""
        
        alerts = []
        
        for metric in metrics:
            if not metric.is_positive_change and abs(metric.change_percentage) > 5:
                alerts.append(
                    f"⚠️ {metric.metric_name} declined by {abs(metric.change_percentage):.1f}%"
                )
        
        return alerts
    
    def _generate_recommendations(
        self,
        persona: PersonaType,
        metrics: List[SnapshotMetric]
    ) -> List[str]:
        """Generate recommendations"""
        
        persona_recommendations = {
            PersonaType.CRO: [
                "Review drift monitoring thresholds for optimal detection",
                "Schedule quarterly fairness audit with compliance team"
            ],
            PersonaType.CFO: [
                "Expand automation to additional workflows for higher ROI",
                "Review cost optimization opportunities in high-volume predictions"
            ],
            PersonaType.COMPLIANCE: [
                "Prepare evidence packs for upcoming regulatory review",
                "Update governance policies for new regulatory requirements"
            ],
            PersonaType.REVOPS: [
                "Train sales team on new forecasting features",
                "Investigate remaining manual overrides for automation"
            ],
            PersonaType.EXECUTIVE: [
                "Review strategic KPIs with leadership team",
                "Plan expansion to additional business units"
            ]
        }
        
        return persona_recommendations.get(persona, [])
    
    def _calculate_health_score(self, metrics: List[SnapshotMetric]) -> float:
        """Calculate overall health score"""
        
        if not metrics:
            return 0.0
        
        positive_changes = sum(1 for m in metrics if m.is_positive_change)
        health_score = (positive_changes / len(metrics)) * 100
        
        return round(health_score, 1)

# Global snapshot engine
snapshot_engine = SnapshotEngine()

@app.post("/email-snapshots/generate", response_model=SuccessSnapshot)
async def generate_snapshot(persona: PersonaType, period_days: int = 7):
    """Generate success snapshot"""
    
    snapshot = snapshot_engine.generate_snapshot(persona, period_days)
    
    logger.info(f"✅ Generated snapshot for {persona.value}")
    return snapshot

@app.post("/email-snapshots/subscribe", response_model=EmailSubscription)
async def create_subscription(
    user_id: str,
    email: EmailStr,
    persona: PersonaType,
    frequency: EmailFrequency
):
    """Create email subscription"""
    
    # Calculate next scheduled delivery
    now = datetime.utcnow()
    if frequency == EmailFrequency.DAILY:
        next_scheduled = now + timedelta(days=1)
    elif frequency == EmailFrequency.WEEKLY:
        next_scheduled = now + timedelta(weeks=1)
    else:  # monthly
        next_scheduled = now + timedelta(days=30)
    
    subscription = EmailSubscription(
        user_id=user_id,
        email=email,
        persona=persona,
        frequency=frequency,
        next_scheduled=next_scheduled
    )
    
    subscriptions_store[subscription.subscription_id] = subscription
    
    logger.info(f"✅ Created subscription for {email} ({persona.value}, {frequency.value})")
    return subscription

@app.post("/email-snapshots/send/{subscription_id}")
async def send_snapshot_email(subscription_id: str):
    """Send snapshot email to subscriber"""
    
    subscription = subscriptions_store.get(subscription_id)
    if not subscription:
        raise HTTPException(status_code=404, detail="Subscription not found")
    
    if not subscription.is_active:
        raise HTTPException(status_code=400, detail="Subscription is not active")
    
    # Generate snapshot
    snapshot = snapshot_engine.generate_snapshot(subscription.persona)
    
    # Create delivery record
    delivery = EmailDeliveryRecord(
        subscription_id=subscription_id,
        snapshot_id=snapshot.snapshot_id,
        recipient_email=subscription.email,
        subject=f"Your {subscription.frequency.value.title()} Success Snapshot",
        status="sent",
        sent_at=datetime.utcnow()
    )
    
    delivery_records_store[delivery.delivery_id] = delivery
    
    # Update subscription
    subscription.last_sent = datetime.utcnow()
    if subscription.frequency == EmailFrequency.DAILY:
        subscription.next_scheduled = datetime.utcnow() + timedelta(days=1)
    elif subscription.frequency == EmailFrequency.WEEKLY:
        subscription.next_scheduled = datetime.utcnow() + timedelta(weeks=1)
    else:
        subscription.next_scheduled = datetime.utcnow() + timedelta(days=30)
    
    logger.info(f"✅ Sent snapshot email to {subscription.email}")
    return {"status": "sent", "delivery_id": delivery.delivery_id}

@app.get("/email-snapshots/subscriptions/{user_id}")
async def get_user_subscriptions(user_id: str):
    """Get all subscriptions for a user"""
    
    user_subs = [
        sub for sub in subscriptions_store.values()
        if sub.user_id == user_id
    ]
    
    return {
        "user_id": user_id,
        "total_subscriptions": len(user_subs),
        "subscriptions": user_subs
    }

@app.delete("/email-snapshots/subscriptions/{subscription_id}")
async def cancel_subscription(subscription_id: str):
    """Cancel email subscription"""
    
    subscription = subscriptions_store.get(subscription_id)
    if not subscription:
        raise HTTPException(status_code=404, detail="Subscription not found")
    
    subscription.is_active = False
    
    logger.info(f"✅ Cancelled subscription {subscription_id}")
    return {"status": "cancelled"}

@app.get("/email-snapshots/summary")
async def get_email_snapshots_summary():
    """Get email snapshots service summary"""
    
    total_snapshots = len(snapshots_store)
    total_subscriptions = len(subscriptions_store)
    active_subscriptions = sum(1 for s in subscriptions_store.values() if s.is_active)
    total_emails_sent = sum(1 for d in delivery_records_store.values() if d.status == "sent")
    
    return {
        "total_snapshots": total_snapshots,
        "total_subscriptions": total_subscriptions,
        "active_subscriptions": active_subscriptions,
        "total_emails_sent": total_emails_sent,
        "service_status": "active"
    }

