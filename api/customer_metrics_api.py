"""
Task 4.4.44: Provide external API for customers to fetch success metrics
- REST API + Swagger for customer access
- Tenant scoped metrics access
"""

from fastapi import FastAPI, HTTPException, Depends, Security
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.openapi.docs import get_swagger_ui_html
from fastapi.openapi.utils import get_openapi
from pydantic import BaseModel, Field
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
from enum import Enum
import uuid
import logging
import jwt

app = FastAPI(
    title="RBIA Customer Metrics API",
    description="External API for customers to access their RBIA success metrics",
    version="1.0.0"
)
logger = logging.getLogger(__name__)

# Security
security = HTTPBearer()
JWT_SECRET = "rbia-customer-api-secret"  # In production, use proper secret management

class MetricCategory(str, Enum):
    ACCURACY = "accuracy"
    PERFORMANCE = "performance"
    ADOPTION = "adoption"
    BUSINESS_IMPACT = "business_impact"
    TRUST_SCORE = "trust_score"
    SLA_COMPLIANCE = "sla_compliance"

class CustomerMetric(BaseModel):
    metric_id: str
    metric_name: str
    metric_category: MetricCategory
    current_value: float
    target_value: Optional[float] = None
    unit: str
    trend: str  # improving, stable, declining
    last_updated: datetime
    description: str

class MetricsResponse(BaseModel):
    tenant_id: str
    metrics: List[CustomerMetric]
    total_count: int
    generated_at: datetime = Field(default_factory=datetime.utcnow)

class TrustScoreResponse(BaseModel):
    tenant_id: str
    overall_trust_score: float
    component_scores: Dict[str, float]
    trust_level: str
    last_updated: datetime
    recommendations: List[str] = Field(default_factory=list)

class AdoptionResponse(BaseModel):
    tenant_id: str
    adoption_rate: float
    active_templates: int
    monthly_executions: int
    user_adoption: float
    growth_trend: str

# Mock data store (in production, would connect to actual metrics services)
customer_metrics_store: Dict[str, List[CustomerMetric]] = {
    "customer_123": [
        CustomerMetric(
            metric_id="acc_001",
            metric_name="Forecast Accuracy",
            metric_category=MetricCategory.ACCURACY,
            current_value=87.5,
            target_value=85.0,
            unit="percentage",
            trend="improving",
            last_updated=datetime.utcnow(),
            description="Accuracy of revenue forecasts compared to actuals"
        ),
        CustomerMetric(
            metric_id="perf_001",
            metric_name="Average Response Time",
            metric_category=MetricCategory.PERFORMANCE,
            current_value=1.2,
            target_value=2.0,
            unit="seconds",
            trend="stable",
            last_updated=datetime.utcnow(),
            description="Average API response time for RBIA workflows"
        ),
        CustomerMetric(
            metric_id="trust_001",
            metric_name="Overall Trust Score",
            metric_category=MetricCategory.TRUST_SCORE,
            current_value=0.89,
            target_value=0.85,
            unit="score",
            trend="improving",
            last_updated=datetime.utcnow(),
            description="Composite trust score across all RBIA capabilities"
        )
    ]
}

def verify_customer_token(credentials: HTTPAuthorizationCredentials = Security(security)) -> str:
    """Verify customer JWT token and return tenant_id"""
    try:
        token = credentials.credentials
        payload = jwt.decode(token, JWT_SECRET, algorithms=["HS256"])
        tenant_id = payload.get("tenant_id")
        
        if not tenant_id:
            raise HTTPException(status_code=401, detail="Invalid token: missing tenant_id")
        
        return tenant_id
    
    except jwt.ExpiredSignatureError:
        raise HTTPException(status_code=401, detail="Token has expired")
    except jwt.InvalidTokenError:
        raise HTTPException(status_code=401, detail="Invalid token")

@app.get("/", include_in_schema=False)
async def custom_swagger_ui_html():
    """Custom Swagger UI for customer API"""
    return get_swagger_ui_html(
        openapi_url="/openapi.json",
        title="RBIA Customer Metrics API - Interactive Documentation",
        swagger_favicon_url="/static/favicon.ico"
    )

@app.get("/openapi.json", include_in_schema=False)
async def get_custom_openapi():
    """Custom OpenAPI schema"""
    if app.openapi_schema:
        return app.openapi_schema
    
    openapi_schema = get_openapi(
        title="RBIA Customer Metrics API",
        version="1.0.0",
        description="""
        ## RBIA Customer Metrics API
        
        This API provides customers with secure access to their RBIA success metrics.
        
        ### Authentication
        All endpoints require a valid JWT token in the Authorization header:
        ```
        Authorization: Bearer <your-jwt-token>
        ```
        
        ### Rate Limits
        - 1000 requests per hour per tenant
        - 100 requests per minute per tenant
        
        ### Support
        For API support, contact: api-support@rbia.com
        """,
        routes=app.routes,
    )
    
    # Add security scheme
    openapi_schema["components"]["securitySchemes"] = {
        "bearerAuth": {
            "type": "http",
            "scheme": "bearer",
            "bearerFormat": "JWT"
        }
    }
    
    # Apply security to all endpoints
    for path in openapi_schema["paths"]:
        for method in openapi_schema["paths"][path]:
            if method != "options":
                openapi_schema["paths"][path][method]["security"] = [{"bearerAuth": []}]
    
    app.openapi_schema = openapi_schema
    return app.openapi_schema

@app.get("/metrics", response_model=MetricsResponse)
async def get_customer_metrics(
    category: Optional[MetricCategory] = None,
    start_date: Optional[datetime] = None,
    end_date: Optional[datetime] = None,
    tenant_id: str = Depends(verify_customer_token)
):
    """
    Get success metrics for the authenticated customer
    
    - **category**: Filter by metric category
    - **start_date**: Filter metrics updated after this date
    - **end_date**: Filter metrics updated before this date
    """
    
    # Get metrics for tenant
    tenant_metrics = customer_metrics_store.get(tenant_id, [])
    
    # Apply filters
    filtered_metrics = tenant_metrics.copy()
    
    if category:
        filtered_metrics = [m for m in filtered_metrics if m.metric_category == category]
    
    if start_date:
        filtered_metrics = [m for m in filtered_metrics if m.last_updated >= start_date]
    
    if end_date:
        filtered_metrics = [m for m in filtered_metrics if m.last_updated <= end_date]
    
    logger.info(f"✅ Served {len(filtered_metrics)} metrics to customer {tenant_id}")
    
    return MetricsResponse(
        tenant_id=tenant_id,
        metrics=filtered_metrics,
        total_count=len(filtered_metrics)
    )

@app.get("/trust-score", response_model=TrustScoreResponse)
async def get_trust_score(tenant_id: str = Depends(verify_customer_token)):
    """
    Get current trust score and component breakdown
    """
    
    # Mock trust score data (would integrate with actual trust score service)
    trust_data = {
        "overall_trust_score": 0.89,
        "component_scores": {
            "accuracy": 0.92,
            "explainability": 0.87,
            "drift_stability": 0.91,
            "bias_fairness": 0.85,
            "sla_compliance": 0.94
        },
        "trust_level": "High",
        "recommendations": [
            "Continue monitoring bias metrics for sustained fairness",
            "Consider expanding explainability coverage to newer models"
        ]
    }
    
    return TrustScoreResponse(
        tenant_id=tenant_id,
        overall_trust_score=trust_data["overall_trust_score"],
        component_scores=trust_data["component_scores"],
        trust_level=trust_data["trust_level"],
        last_updated=datetime.utcnow(),
        recommendations=trust_data["recommendations"]
    )

@app.get("/adoption", response_model=AdoptionResponse)
async def get_adoption_metrics(tenant_id: str = Depends(verify_customer_token)):
    """
    Get RBIA adoption metrics for the customer
    """
    
    # Mock adoption data (would integrate with actual adoption service)
    adoption_data = {
        "adoption_rate": 78.5,
        "active_templates": 12,
        "monthly_executions": 1450,
        "user_adoption": 82.3,
        "growth_trend": "improving"
    }
    
    return AdoptionResponse(
        tenant_id=tenant_id,
        **adoption_data
    )

@app.get("/sla-compliance")
async def get_sla_compliance(tenant_id: str = Depends(verify_customer_token)):
    """
    Get SLA compliance metrics
    """
    
    # Mock SLA data (would integrate with actual SLA monitoring service)
    sla_data = {
        "tenant_id": tenant_id,
        "overall_compliance": 98.7,
        "availability": 99.9,
        "average_response_time_ms": 1200,
        "p95_response_time_ms": 2100,
        "error_rate": 0.02,
        "sla_breaches_this_month": 1,
        "uptime_percentage": 99.95,
        "last_updated": datetime.utcnow().isoformat()
    }
    
    return sla_data

@app.get("/business-impact")
async def get_business_impact(tenant_id: str = Depends(verify_customer_token)):
    """
    Get business impact metrics
    """
    
    # Mock business impact data
    impact_data = {
        "tenant_id": tenant_id,
        "roi_percentage": 245.7,
        "cost_savings_monthly": 125000.0,
        "revenue_impact_monthly": 89000.0,
        "efficiency_gain_percentage": 34.2,
        "time_saved_hours_monthly": 890,
        "error_reduction_percentage": 67.8,
        "last_calculated": datetime.utcnow().isoformat()
    }
    
    return impact_data

@app.get("/export/csv")
async def export_metrics_csv(
    category: Optional[MetricCategory] = None,
    tenant_id: str = Depends(verify_customer_token)
):
    """
    Export metrics as CSV format
    """
    
    # Get metrics
    tenant_metrics = customer_metrics_store.get(tenant_id, [])
    
    if category:
        tenant_metrics = [m for m in tenant_metrics if m.metric_category == category]
    
    # Generate CSV content
    csv_content = "metric_name,category,current_value,target_value,unit,trend,last_updated\n"
    
    for metric in tenant_metrics:
        csv_content += f"{metric.metric_name},{metric.metric_category.value},{metric.current_value},{metric.target_value or ''},"
        csv_content += f"{metric.unit},{metric.trend},{metric.last_updated.isoformat()}\n"
    
    from fastapi.responses import Response
    
    return Response(
        content=csv_content,
        media_type="text/csv",
        headers={"Content-Disposition": f"attachment; filename=rbia_metrics_{tenant_id}.csv"}
    )

@app.get("/health")
async def health_check():
    """
    API health check endpoint (public)
    """
    return {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "version": "1.0.0"
    }

# Utility endpoint for generating customer tokens (for demo purposes)
@app.post("/auth/demo-token")
async def generate_demo_token(tenant_id: str):
    """
    Generate a demo JWT token for testing (remove in production)
    """
    
    payload = {
        "tenant_id": tenant_id,
        "exp": datetime.utcnow() + timedelta(hours=24),
        "iat": datetime.utcnow()
    }
    
    token = jwt.encode(payload, JWT_SECRET, algorithm="HS256")
    
    return {
        "access_token": token,
        "token_type": "bearer",
        "expires_in": 86400,
        "tenant_id": tenant_id
    }

@app.get("/api-info")
async def get_api_info():
    """
    Get API information and usage guidelines
    """
    return {
        "api_name": "RBIA Customer Metrics API",
        "version": "1.0.0",
        "description": "Secure access to RBIA success metrics for customers",
        "authentication": "JWT Bearer Token",
        "rate_limits": {
            "per_hour": 1000,
            "per_minute": 100
        },
        "supported_formats": ["JSON", "CSV"],
        "documentation_url": "/docs",
        "support_email": "api-support@rbia.com"
    }
