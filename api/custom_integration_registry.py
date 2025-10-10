"""
Task 4.5.37: Provide customer-facing APIs to register custom integrations
- Self-service integration registration
- Custom integration configuration
- Integration testing endpoints
- Integration validation framework
- Tenant-isolated custom integrations
"""

from fastapi import FastAPI, HTTPException, UploadFile, File
from pydantic import BaseModel, Field, HttpUrl
from typing import Dict, List, Any, Optional
from datetime import datetime
from enum import Enum
import uuid
import logging
import json

app = FastAPI(title="RBIA Custom Integration Registry")
logger = logging.getLogger(__name__)

class IntegrationMethod(str, Enum):
    REST_API = "rest_api"
    WEBHOOK = "webhook"
    GRAPHQL = "graphql"
    GRPC = "grpc"
    CUSTOM = "custom"

class AuthenticationType(str, Enum):
    API_KEY = "api_key"
    OAUTH2 = "oauth2"
    BASIC_AUTH = "basic_auth"
    BEARER_TOKEN = "bearer_token"
    CUSTOM = "custom"

class IntegrationStatus(str, Enum):
    DRAFT = "draft"
    TESTING = "testing"
    VALIDATED = "validated"
    ACTIVE = "active"
    INACTIVE = "inactive"
    FAILED = "failed"

class DataDirection(str, Enum):
    INBOUND = "inbound"  # External system → RBIA
    OUTBOUND = "outbound"  # RBIA → External system
    BIDIRECTIONAL = "bidirectional"

class CustomIntegration(BaseModel):
    integration_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    tenant_id: str
    created_by_user_id: str
    
    # Basic information
    integration_name: str
    description: str
    integration_category: str  # crm, billing, analytics, etc.
    
    # Technical configuration
    integration_method: IntegrationMethod
    base_url: HttpUrl
    authentication_type: AuthenticationType
    
    # Data flow
    data_direction: DataDirection
    
    # Endpoints configuration
    endpoints: List[Dict[str, Any]] = Field(default_factory=list)
    
    # Data mapping
    field_mappings: Dict[str, str] = Field(default_factory=dict)
    
    # Behavior configuration
    retry_policy: Dict[str, Any] = Field(default_factory=dict)
    timeout_seconds: int = 30
    rate_limit_per_minute: int = 60
    
    # Status
    status: IntegrationStatus = IntegrationStatus.DRAFT
    is_validated: bool = False
    validation_errors: List[str] = Field(default_factory=list)
    
    # Testing
    test_endpoint_available: bool = False
    last_test_at: Optional[datetime] = None
    last_test_success: bool = False
    
    # Usage tracking
    total_calls: int = 0
    successful_calls: int = 0
    failed_calls: int = 0
    
    # Metadata
    tags: List[str] = Field(default_factory=list)
    documentation_url: Optional[HttpUrl] = None
    support_contact: Optional[str] = None
    
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)

class EndpointConfiguration(BaseModel):
    endpoint_name: str
    endpoint_path: str
    http_method: str = "POST"
    description: str
    
    # Request configuration
    request_headers: Dict[str, str] = Field(default_factory=dict)
    request_body_template: Dict[str, Any] = Field(default_factory=dict)
    
    # Response configuration
    response_mapping: Dict[str, str] = Field(default_factory=dict)
    success_status_codes: List[int] = Field(default=[200, 201, 202])
    
    # Error handling
    error_handling: Dict[str, Any] = Field(default_factory=dict)

class IntegrationTest(BaseModel):
    test_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    integration_id: str
    tenant_id: str
    
    # Test configuration
    test_type: str  # connectivity, authentication, data_flow, end_to_end
    test_payload: Dict[str, Any] = Field(default_factory=dict)
    
    # Results
    success: bool = False
    response_time_ms: float = 0.0
    response_data: Optional[Dict[str, Any]] = None
    error_message: Optional[str] = None
    
    # Validation results
    validation_checks: List[Dict[str, Any]] = Field(default_factory=list)
    
    executed_at: datetime = Field(default_factory=datetime.utcnow)

class IntegrationValidation(BaseModel):
    validation_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    integration_id: str
    tenant_id: str
    
    # Validation checks
    checks_performed: List[str] = Field(default_factory=list)
    checks_passed: List[str] = Field(default_factory=list)
    checks_failed: List[str] = Field(default_factory=list)
    
    # Overall result
    overall_status: str  # passed, failed, partial
    validation_score: float = 0.0
    
    # Detailed findings
    findings: List[Dict[str, Any]] = Field(default_factory=list)
    recommendations: List[str] = Field(default_factory=list)
    
    validated_at: datetime = Field(default_factory=datetime.utcnow)

class IntegrationCatalog(BaseModel):
    catalog_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    tenant_id: str
    
    # Catalog items
    integrations: List[CustomIntegration] = Field(default_factory=list)
    
    # Statistics
    total_integrations: int = 0
    active_integrations: int = 0
    validated_integrations: int = 0
    
    # Categories
    categories: Dict[str, int] = Field(default_factory=dict)
    
    generated_at: datetime = Field(default_factory=datetime.utcnow)

# In-memory storage
integrations_store: Dict[str, CustomIntegration] = {}
tests_store: Dict[str, IntegrationTest] = {}
validations_store: Dict[str, IntegrationValidation] = {}

class CustomIntegrationService:
    """Service for managing custom integrations"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    async def validate_integration(
        self,
        integration: CustomIntegration
    ) -> IntegrationValidation:
        """Validate custom integration configuration"""
        
        checks = [
            "url_format",
            "authentication_config",
            "endpoint_configuration",
            "field_mappings",
            "retry_policy",
            "rate_limits",
            "security_compliance"
        ]
        
        passed = []
        failed = []
        findings = []
        
        # Check URL format
        if integration.base_url:
            passed.append("url_format")
            findings.append({
                "check": "url_format",
                "status": "passed",
                "message": "Base URL is properly formatted"
            })
        else:
            failed.append("url_format")
            findings.append({
                "check": "url_format",
                "status": "failed",
                "message": "Base URL is required"
            })
        
        # Check authentication
        if integration.authentication_type:
            passed.append("authentication_config")
            findings.append({
                "check": "authentication_config",
                "status": "passed",
                "message": f"Authentication type configured: {integration.authentication_type}"
            })
        else:
            failed.append("authentication_config")
        
        # Check endpoints
        if len(integration.endpoints) > 0:
            passed.append("endpoint_configuration")
            findings.append({
                "check": "endpoint_configuration",
                "status": "passed",
                "message": f"{len(integration.endpoints)} endpoints configured"
            })
        else:
            failed.append("endpoint_configuration")
            findings.append({
                "check": "endpoint_configuration",
                "status": "failed",
                "message": "At least one endpoint must be configured"
            })
        
        # Check field mappings
        if len(integration.field_mappings) > 0:
            passed.append("field_mappings")
        else:
            findings.append({
                "check": "field_mappings",
                "status": "warning",
                "message": "No field mappings configured (optional but recommended)"
            })
        
        # Check retry policy
        if integration.retry_policy:
            passed.append("retry_policy")
        else:
            findings.append({
                "check": "retry_policy",
                "status": "warning",
                "message": "No retry policy configured (will use default)"
            })
        
        # Check rate limits
        if integration.rate_limit_per_minute > 0:
            passed.append("rate_limits")
        
        # Security compliance check
        if integration.authentication_type != AuthenticationType.CUSTOM:
            passed.append("security_compliance")
            findings.append({
                "check": "security_compliance",
                "status": "passed",
                "message": "Using standard authentication method"
            })
        
        # Calculate validation score
        validation_score = (len(passed) / len(checks)) * 100
        
        # Determine overall status
        if len(failed) == 0:
            overall_status = "passed"
        elif len(passed) >= len(checks) / 2:
            overall_status = "partial"
        else:
            overall_status = "failed"
        
        # Recommendations
        recommendations = []
        if "endpoint_configuration" in failed:
            recommendations.append("Add at least one endpoint configuration")
        if "field_mappings" not in passed:
            recommendations.append("Define field mappings for data transformation")
        if "retry_policy" not in passed:
            recommendations.append("Configure retry policy for resilience")
        
        validation = IntegrationValidation(
            integration_id=integration.integration_id,
            tenant_id=integration.tenant_id,
            checks_performed=checks,
            checks_passed=passed,
            checks_failed=failed,
            overall_status=overall_status,
            validation_score=validation_score,
            findings=findings,
            recommendations=recommendations
        )
        
        return validation
    
    async def test_integration(
        self,
        integration_id: str,
        test_type: str,
        test_payload: Dict[str, Any]
    ) -> IntegrationTest:
        """Test custom integration"""
        
        if integration_id not in integrations_store:
            raise ValueError(f"Integration {integration_id} not found")
        
        integration = integrations_store[integration_id]
        
        # Simulate test execution
        test = IntegrationTest(
            integration_id=integration_id,
            tenant_id=integration.tenant_id,
            test_type=test_type,
            test_payload=test_payload
        )
        
        # Mock test results
        import random
        test.success = random.choice([True, True, True, False])  # 75% success rate
        test.response_time_ms = random.uniform(100, 1000)
        
        if test.success:
            test.response_data = {
                "status": "success",
                "message": "Test completed successfully",
                "data": {"mock": "response"}
            }
            
            test.validation_checks = [
                {"check": "connectivity", "passed": True},
                {"check": "authentication", "passed": True},
                {"check": "response_format", "passed": True}
            ]
        else:
            test.error_message = "Connection timeout"
            test.validation_checks = [
                {"check": "connectivity", "passed": False, "error": "Timeout after 30s"}
            ]
        
        # Update integration test tracking
        integration.last_test_at = datetime.utcnow()
        integration.last_test_success = test.success
        
        tests_store[test.test_id] = test
        return test

integration_service = CustomIntegrationService()

@app.post("/custom-integrations", response_model=CustomIntegration)
async def register_custom_integration(integration: CustomIntegration):
    """Register a new custom integration"""
    
    # Validate tenant isolation
    if not integration.tenant_id:
        raise HTTPException(status_code=400, detail="tenant_id is required")
    
    # Store integration
    integrations_store[integration.integration_id] = integration
    
    logger.info(f"Registered custom integration: {integration.integration_name} for tenant {integration.tenant_id}")
    
    return integration

@app.get("/custom-integrations", response_model=List[CustomIntegration])
async def list_custom_integrations(
    tenant_id: str,
    status: Optional[IntegrationStatus] = None,
    category: Optional[str] = None
):
    """List custom integrations for a tenant"""
    
    integrations = [i for i in integrations_store.values() if i.tenant_id == tenant_id]
    
    if status:
        integrations = [i for i in integrations if i.status == status]
    
    if category:
        integrations = [i for i in integrations if i.integration_category == category]
    
    return integrations

@app.get("/custom-integrations/{integration_id}", response_model=CustomIntegration)
async def get_custom_integration(integration_id: str, tenant_id: str):
    """Get custom integration details"""
    
    if integration_id not in integrations_store:
        raise HTTPException(status_code=404, detail="Integration not found")
    
    integration = integrations_store[integration_id]
    
    # Enforce tenant isolation
    if integration.tenant_id != tenant_id:
        raise HTTPException(status_code=403, detail="Access denied")
    
    return integration

@app.put("/custom-integrations/{integration_id}", response_model=CustomIntegration)
async def update_custom_integration(
    integration_id: str,
    tenant_id: str,
    updates: Dict[str, Any]
):
    """Update custom integration"""
    
    if integration_id not in integrations_store:
        raise HTTPException(status_code=404, detail="Integration not found")
    
    integration = integrations_store[integration_id]
    
    # Enforce tenant isolation
    if integration.tenant_id != tenant_id:
        raise HTTPException(status_code=403, detail="Access denied")
    
    # Apply updates
    for key, value in updates.items():
        if hasattr(integration, key):
            setattr(integration, key, value)
    
    integration.updated_at = datetime.utcnow()
    
    logger.info(f"Updated custom integration: {integration_id}")
    
    return integration

@app.post("/custom-integrations/{integration_id}/validate", response_model=IntegrationValidation)
async def validate_custom_integration(integration_id: str, tenant_id: str):
    """Validate custom integration configuration"""
    
    if integration_id not in integrations_store:
        raise HTTPException(status_code=404, detail="Integration not found")
    
    integration = integrations_store[integration_id]
    
    # Enforce tenant isolation
    if integration.tenant_id != tenant_id:
        raise HTTPException(status_code=403, detail="Access denied")
    
    try:
        validation = await integration_service.validate_integration(integration)
        
        # Update integration status
        integration.is_validated = validation.overall_status == "passed"
        integration.validation_errors = [f["message"] for f in validation.findings if f["status"] == "failed"]
        
        if integration.is_validated:
            integration.status = IntegrationStatus.VALIDATED
        
        validations_store[validation.validation_id] = validation
        
        return validation
        
    except Exception as e:
        logger.error(f"Validation failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/custom-integrations/{integration_id}/test", response_model=IntegrationTest)
async def test_custom_integration(
    integration_id: str,
    tenant_id: str,
    test_type: str = "connectivity",
    test_payload: Dict[str, Any] = {}
):
    """Test custom integration"""
    
    try:
        test = await integration_service.test_integration(
            integration_id,
            test_type,
            test_payload
        )
        
        return test
        
    except Exception as e:
        logger.error(f"Test failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/custom-integrations/{integration_id}/activate")
async def activate_custom_integration(integration_id: str, tenant_id: str):
    """Activate custom integration"""
    
    if integration_id not in integrations_store:
        raise HTTPException(status_code=404, detail="Integration not found")
    
    integration = integrations_store[integration_id]
    
    # Enforce tenant isolation
    if integration.tenant_id != tenant_id:
        raise HTTPException(status_code=403, detail="Access denied")
    
    # Check if validated
    if not integration.is_validated:
        raise HTTPException(status_code=400, detail="Integration must be validated before activation")
    
    integration.status = IntegrationStatus.ACTIVE
    integration.updated_at = datetime.utcnow()
    
    logger.info(f"Activated custom integration: {integration_id}")
    
    return {
        "status": "activated",
        "integration_id": integration_id,
        "activated_at": datetime.utcnow().isoformat()
    }

@app.post("/custom-integrations/{integration_id}/deactivate")
async def deactivate_custom_integration(integration_id: str, tenant_id: str):
    """Deactivate custom integration"""
    
    if integration_id not in integrations_store:
        raise HTTPException(status_code=404, detail="Integration not found")
    
    integration = integrations_store[integration_id]
    
    # Enforce tenant isolation
    if integration.tenant_id != tenant_id:
        raise HTTPException(status_code=403, detail="Access denied")
    
    integration.status = IntegrationStatus.INACTIVE
    integration.updated_at = datetime.utcnow()
    
    logger.info(f"Deactivated custom integration: {integration_id}")
    
    return {
        "status": "deactivated",
        "integration_id": integration_id,
        "deactivated_at": datetime.utcnow().isoformat()
    }

@app.delete("/custom-integrations/{integration_id}")
async def delete_custom_integration(integration_id: str, tenant_id: str):
    """Delete custom integration"""
    
    if integration_id not in integrations_store:
        raise HTTPException(status_code=404, detail="Integration not found")
    
    integration = integrations_store[integration_id]
    
    # Enforce tenant isolation
    if integration.tenant_id != tenant_id:
        raise HTTPException(status_code=403, detail="Access denied")
    
    del integrations_store[integration_id]
    
    logger.info(f"Deleted custom integration: {integration_id}")
    
    return {"status": "deleted", "integration_id": integration_id}

@app.get("/custom-integrations/catalog/{tenant_id}", response_model=IntegrationCatalog)
async def get_integration_catalog(tenant_id: str):
    """Get integration catalog for tenant"""
    
    integrations = [i for i in integrations_store.values() if i.tenant_id == tenant_id]
    
    # Calculate statistics
    total = len(integrations)
    active = sum(1 for i in integrations if i.status == IntegrationStatus.ACTIVE)
    validated = sum(1 for i in integrations if i.is_validated)
    
    # Count by category
    categories = {}
    for integration in integrations:
        cat = integration.integration_category
        categories[cat] = categories.get(cat, 0) + 1
    
    catalog = IntegrationCatalog(
        tenant_id=tenant_id,
        integrations=integrations,
        total_integrations=total,
        active_integrations=active,
        validated_integrations=validated,
        categories=categories
    )
    
    return catalog

@app.post("/custom-integrations/import")
async def import_integration_config(
    tenant_id: str,
    file: UploadFile = File(...)
):
    """Import integration configuration from JSON file"""
    
    try:
        content = await file.read()
        config = json.loads(content)
        
        # Create integration from config
        integration = CustomIntegration(**config, tenant_id=tenant_id)
        integrations_store[integration.integration_id] = integration
        
        logger.info(f"Imported integration config: {integration.integration_name}")
        
        return {
            "status": "imported",
            "integration_id": integration.integration_id,
            "integration_name": integration.integration_name
        }
        
    except Exception as e:
        logger.error(f"Import failed: {e}")
        raise HTTPException(status_code=400, detail=f"Import failed: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8037)

