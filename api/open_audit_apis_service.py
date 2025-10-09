"""
Task 3.4.36: Develop open audit APIs
- REST API endpoints for evidence export
- Authentication and authorization for audit access
- Data serialization for multiple export formats (JSON, PDF, CSV)
- API rate limiting and tenant isolation
"""

from fastapi import FastAPI, HTTPException, Depends, Request, BackgroundTasks
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.responses import StreamingResponse, FileResponse
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any, Union
from datetime import datetime, timedelta
from enum import Enum
import uuid
import logging
import json
import csv
import io
import zipfile
from dataclasses import dataclass
import time
import hashlib

app = FastAPI(title="RBIA Open Audit APIs")
logger = logging.getLogger(__name__)
security = HTTPBearer()

class ExportFormat(str, Enum):
    JSON = "json"
    CSV = "csv"
    PDF = "pdf"
    XML = "xml"
    ZIP = "zip"

class AuditDataType(str, Enum):
    EXECUTION_TRACES = "execution_traces"
    OVERRIDE_LEDGER = "override_ledger"
    EVIDENCE_PACKS = "evidence_packs"
    POLICY_VIOLATIONS = "policy_violations"
    TRUST_SCORES = "trust_scores"
    COMPLIANCE_REPORTS = "compliance_reports"
    WORKFLOW_DEFINITIONS = "workflow_definitions"
    USER_ACTIVITIES = "user_activities"

class AccessLevel(str, Enum):
    READ_ONLY = "read_only"
    EXPORT_BASIC = "export_basic"
    EXPORT_DETAILED = "export_detailed"
    FULL_AUDIT = "full_audit"

class AuditExportRequest(BaseModel):
    export_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    tenant_id: str
    data_types: List[AuditDataType]
    export_format: ExportFormat
    
    # Filters
    date_from: Optional[datetime] = None
    date_to: Optional[datetime] = None
    workflow_ids: Optional[List[str]] = None
    user_ids: Optional[List[str]] = None
    
    # Export options
    include_metadata: bool = True
    include_raw_data: bool = False
    anonymize_pii: bool = True
    
    # Delivery
    callback_url: Optional[str] = None
    email_notification: Optional[str] = None
    
    requested_by: str
    requested_at: datetime = Field(default_factory=datetime.utcnow)

class AuditExportStatus(BaseModel):
    export_id: str
    status: str  # pending, processing, completed, failed, expired
    progress_percentage: float = 0.0
    
    # File details
    file_size_bytes: Optional[int] = None
    download_url: Optional[str] = None
    expires_at: Optional[datetime] = None
    
    # Metadata
    records_exported: int = 0
    processing_time_ms: Optional[int] = None
    error_message: Optional[str] = None
    
    created_at: datetime = Field(default_factory=datetime.utcnow)
    completed_at: Optional[datetime] = None

class APIKey(BaseModel):
    key_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    tenant_id: str
    key_name: str
    
    # Access control
    access_level: AccessLevel
    allowed_data_types: List[AuditDataType] = Field(default_factory=list)
    allowed_formats: List[ExportFormat] = Field(default_factory=list)
    
    # Rate limiting
    rate_limit_per_hour: int = 100
    rate_limit_per_day: int = 1000
    
    # Security
    api_key_hash: str  # SHA-256 hash of the actual key
    last_used_at: Optional[datetime] = None
    
    # Lifecycle
    created_at: datetime = Field(default_factory=datetime.utcnow)
    expires_at: Optional[datetime] = None
    is_active: bool = True
    
    created_by: str

class RateLimitInfo(BaseModel):
    requests_remaining: int
    reset_time: datetime
    limit_type: str  # hourly, daily

# In-memory storage (replace with database in production)
export_requests_store: Dict[str, AuditExportRequest] = {}
export_status_store: Dict[str, AuditExportStatus] = {}
api_keys_store: Dict[str, APIKey] = {}
rate_limit_store: Dict[str, Dict[str, Any]] = {}  # key_id -> {hourly_count, daily_count, last_reset}

# Authentication and authorization
async def verify_api_key(credentials: HTTPAuthorizationCredentials = Depends(security)) -> APIKey:
    """Verify API key and return associated permissions"""
    
    api_key_hash = hashlib.sha256(credentials.credentials.encode()).hexdigest()
    
    # Find matching API key
    for api_key in api_keys_store.values():
        if api_key.api_key_hash == api_key_hash and api_key.is_active:
            # Check expiration
            if api_key.expires_at and api_key.expires_at < datetime.utcnow():
                raise HTTPException(status_code=401, detail="API key expired")
            
            # Update last used
            api_key.last_used_at = datetime.utcnow()
            api_keys_store[api_key.key_id] = api_key
            
            return api_key
    
    raise HTTPException(status_code=401, detail="Invalid API key")

async def check_rate_limit(api_key: APIKey) -> RateLimitInfo:
    """Check and update rate limits"""
    
    now = datetime.utcnow()
    key_id = api_key.key_id
    
    # Initialize rate limit tracking
    if key_id not in rate_limit_store:
        rate_limit_store[key_id] = {
            "hourly_count": 0,
            "daily_count": 0,
            "last_hourly_reset": now,
            "last_daily_reset": now
        }
    
    limits = rate_limit_store[key_id]
    
    # Reset hourly counter
    if now - limits["last_hourly_reset"] >= timedelta(hours=1):
        limits["hourly_count"] = 0
        limits["last_hourly_reset"] = now
    
    # Reset daily counter
    if now - limits["last_daily_reset"] >= timedelta(days=1):
        limits["daily_count"] = 0
        limits["last_daily_reset"] = now
    
    # Check limits
    if limits["hourly_count"] >= api_key.rate_limit_per_hour:
        raise HTTPException(
            status_code=429, 
            detail=f"Hourly rate limit exceeded ({api_key.rate_limit_per_hour} requests/hour)"
        )
    
    if limits["daily_count"] >= api_key.rate_limit_per_day:
        raise HTTPException(
            status_code=429, 
            detail=f"Daily rate limit exceeded ({api_key.rate_limit_per_day} requests/day)"
        )
    
    # Increment counters
    limits["hourly_count"] += 1
    limits["daily_count"] += 1
    rate_limit_store[key_id] = limits
    
    return RateLimitInfo(
        requests_remaining=min(
            api_key.rate_limit_per_hour - limits["hourly_count"],
            api_key.rate_limit_per_day - limits["daily_count"]
        ),
        reset_time=min(
            limits["last_hourly_reset"] + timedelta(hours=1),
            limits["last_daily_reset"] + timedelta(days=1)
        ),
        limit_type="hourly" if api_key.rate_limit_per_hour - limits["hourly_count"] < api_key.rate_limit_per_day - limits["daily_count"] else "daily"
    )

@app.post("/audit/export", response_model=AuditExportStatus)
async def create_export_request(
    request: AuditExportRequest,
    background_tasks: BackgroundTasks,
    api_key: APIKey = Depends(verify_api_key)
):
    """Create audit data export request"""
    
    # Check rate limits
    await check_rate_limit(api_key)
    
    # Validate tenant access
    if api_key.tenant_id != request.tenant_id:
        raise HTTPException(status_code=403, detail="Access denied to tenant data")
    
    # Validate data type permissions
    for data_type in request.data_types:
        if data_type not in api_key.allowed_data_types:
            raise HTTPException(status_code=403, detail=f"Access denied to {data_type.value}")
    
    # Validate format permissions
    if request.export_format not in api_key.allowed_formats:
        raise HTTPException(status_code=403, detail=f"Export format {request.export_format.value} not allowed")
    
    # Create export status
    export_status = AuditExportStatus(
        export_id=request.export_id,
        status="pending"
    )
    
    # Store request and status
    export_requests_store[request.export_id] = request
    export_status_store[request.export_id] = export_status
    
    # Process export in background
    background_tasks.add_task(_process_export_request, request, api_key)
    
    logger.info(f"📤 Created export request {request.export_id} for tenant {request.tenant_id}")
    return export_status

@app.get("/audit/export/{export_id}/status", response_model=AuditExportStatus)
async def get_export_status(
    export_id: str,
    api_key: APIKey = Depends(verify_api_key)
):
    """Get export request status"""
    
    if export_id not in export_status_store:
        raise HTTPException(status_code=404, detail="Export request not found")
    
    export_status = export_status_store[export_id]
    
    # Validate tenant access
    request = export_requests_store.get(export_id)
    if request and api_key.tenant_id != request.tenant_id:
        raise HTTPException(status_code=403, detail="Access denied")
    
    return export_status

@app.get("/audit/export/{export_id}/download")
async def download_export(
    export_id: str,
    api_key: APIKey = Depends(verify_api_key)
):
    """Download completed export"""
    
    if export_id not in export_status_store:
        raise HTTPException(status_code=404, detail="Export request not found")
    
    export_status = export_status_store[export_id]
    request = export_requests_store[export_id]
    
    # Validate access
    if api_key.tenant_id != request.tenant_id:
        raise HTTPException(status_code=403, detail="Access denied")
    
    if export_status.status != "completed":
        raise HTTPException(status_code=400, detail="Export not completed")
    
    if export_status.expires_at and export_status.expires_at < datetime.utcnow():
        raise HTTPException(status_code=410, detail="Export expired")
    
    # Generate file for download
    file_content = await _generate_export_file(request)
    
    # Determine content type
    content_type_map = {
        ExportFormat.JSON: "application/json",
        ExportFormat.CSV: "text/csv",
        ExportFormat.XML: "application/xml",
        ExportFormat.ZIP: "application/zip"
    }
    
    filename = f"audit_export_{export_id}.{request.export_format.value}"
    
    return StreamingResponse(
        io.BytesIO(file_content),
        media_type=content_type_map.get(request.export_format, "application/octet-stream"),
        headers={"Content-Disposition": f"attachment; filename={filename}"}
    )

@app.get("/audit/data-types", response_model=List[Dict[str, Any]])
async def list_available_data_types(api_key: APIKey = Depends(verify_api_key)):
    """List available audit data types for the API key"""
    
    data_type_info = []
    
    for data_type in api_key.allowed_data_types:
        info = {
            "data_type": data_type.value,
            "description": _get_data_type_description(data_type),
            "estimated_records": _get_estimated_record_count(data_type, api_key.tenant_id),
            "retention_days": _get_retention_period(data_type),
            "available_formats": [fmt.value for fmt in api_key.allowed_formats]
        }
        data_type_info.append(info)
    
    return data_type_info

@app.get("/audit/exports", response_model=List[AuditExportStatus])
async def list_exports(
    api_key: APIKey = Depends(verify_api_key),
    status: Optional[str] = None,
    limit: int = 50
):
    """List export requests for the tenant"""
    
    # Filter exports by tenant
    tenant_exports = []
    for export_id, export_status in export_status_store.items():
        request = export_requests_store.get(export_id)
        if request and request.tenant_id == api_key.tenant_id:
            if not status or export_status.status == status:
                tenant_exports.append(export_status)
    
    # Sort by creation date (newest first)
    tenant_exports.sort(key=lambda x: x.created_at, reverse=True)
    
    return tenant_exports[:limit]

@app.delete("/audit/export/{export_id}")
async def cancel_export(
    export_id: str,
    api_key: APIKey = Depends(verify_api_key)
):
    """Cancel or delete export request"""
    
    if export_id not in export_status_store:
        raise HTTPException(status_code=404, detail="Export request not found")
    
    request = export_requests_store[export_id]
    if api_key.tenant_id != request.tenant_id:
        raise HTTPException(status_code=403, detail="Access denied")
    
    export_status = export_status_store[export_id]
    
    if export_status.status in ["pending", "processing"]:
        export_status.status = "cancelled"
        export_status_store[export_id] = export_status
        return {"message": "Export cancelled"}
    elif export_status.status == "completed":
        # Delete completed export
        del export_requests_store[export_id]
        del export_status_store[export_id]
        return {"message": "Export deleted"}
    else:
        raise HTTPException(status_code=400, detail="Cannot cancel export in current status")

@app.post("/audit/api-keys", response_model=Dict[str, str])
async def create_api_key(
    tenant_id: str,
    key_name: str,
    access_level: AccessLevel,
    allowed_data_types: List[AuditDataType],
    allowed_formats: List[ExportFormat],
    created_by: str,
    expires_days: Optional[int] = None
):
    """Create new API key (admin endpoint)"""
    
    # Generate secure API key
    api_key_value = f"rbia_audit_{uuid.uuid4().hex}"
    api_key_hash = hashlib.sha256(api_key_value.encode()).hexdigest()
    
    # Set expiration
    expires_at = None
    if expires_days:
        expires_at = datetime.utcnow() + timedelta(days=expires_days)
    
    # Create API key record
    api_key = APIKey(
        tenant_id=tenant_id,
        key_name=key_name,
        access_level=access_level,
        allowed_data_types=allowed_data_types,
        allowed_formats=allowed_formats,
        api_key_hash=api_key_hash,
        expires_at=expires_at,
        created_by=created_by
    )
    
    api_keys_store[api_key.key_id] = api_key
    
    logger.info(f"🔑 Created API key {api_key.key_id} for tenant {tenant_id}")
    
    return {
        "key_id": api_key.key_id,
        "api_key": api_key_value,  # Only returned once
        "message": "Store this API key securely - it cannot be retrieved again"
    }

async def _process_export_request(request: AuditExportRequest, api_key: APIKey):
    """Process export request in background"""
    
    export_status = export_status_store[request.export_id]
    
    try:
        export_status.status = "processing"
        export_status_store[request.export_id] = export_status
        
        start_time = time.time()
        
        # Simulate data collection and processing
        total_records = 0
        for i, data_type in enumerate(request.data_types):
            # Update progress
            progress = (i / len(request.data_types)) * 100
            export_status.progress_percentage = progress
            export_status_store[request.export_id] = export_status
            
            # Simulate processing time
            await asyncio.sleep(2)
            
            # Count records (simulated)
            records = _get_estimated_record_count(data_type, request.tenant_id)
            total_records += records
        
        # Generate export file
        file_content = await _generate_export_file(request)
        
        # Complete export
        processing_time = int((time.time() - start_time) * 1000)
        
        export_status.status = "completed"
        export_status.progress_percentage = 100.0
        export_status.records_exported = total_records
        export_status.file_size_bytes = len(file_content)
        export_status.processing_time_ms = processing_time
        export_status.download_url = f"/audit/export/{request.export_id}/download"
        export_status.expires_at = datetime.utcnow() + timedelta(days=7)  # 7-day download window
        export_status.completed_at = datetime.utcnow()
        
        export_status_store[request.export_id] = export_status
        
        logger.info(f"✅ Completed export {request.export_id} - {total_records} records, {len(file_content)} bytes")
        
    except Exception as e:
        export_status.status = "failed"
        export_status.error_message = str(e)
        export_status_store[request.export_id] = export_status
        logger.error(f"❌ Export {request.export_id} failed: {e}")

async def _generate_export_file(request: AuditExportRequest) -> bytes:
    """Generate export file based on request"""
    
    # Collect data (simulated - replace with actual data queries)
    export_data = {}
    
    for data_type in request.data_types:
        export_data[data_type.value] = _simulate_audit_data(data_type, request)
    
    # Format data based on export format
    if request.export_format == ExportFormat.JSON:
        return json.dumps(export_data, indent=2, default=str).encode('utf-8')
    
    elif request.export_format == ExportFormat.CSV:
        output = io.StringIO()
        
        # Create CSV for each data type
        for data_type, records in export_data.items():
            output.write(f"\n# {data_type.upper()}\n")
            if records:
                writer = csv.DictWriter(output, fieldnames=records[0].keys())
                writer.writeheader()
                writer.writerows(records)
        
        return output.getvalue().encode('utf-8')
    
    elif request.export_format == ExportFormat.ZIP:
        zip_buffer = io.BytesIO()
        
        with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
            # Add JSON file for each data type
            for data_type, records in export_data.items():
                json_content = json.dumps(records, indent=2, default=str)
                zip_file.writestr(f"{data_type}.json", json_content)
            
            # Add metadata file
            metadata = {
                "export_id": request.export_id,
                "tenant_id": request.tenant_id,
                "exported_at": datetime.utcnow().isoformat(),
                "data_types": [dt.value for dt in request.data_types],
                "filters_applied": {
                    "date_from": request.date_from.isoformat() if request.date_from else None,
                    "date_to": request.date_to.isoformat() if request.date_to else None,
                    "anonymize_pii": request.anonymize_pii
                }
            }
            zip_file.writestr("metadata.json", json.dumps(metadata, indent=2))
        
        return zip_buffer.getvalue()
    
    else:
        # Default to JSON
        return json.dumps(export_data, indent=2, default=str).encode('utf-8')

def _simulate_audit_data(data_type: AuditDataType, request: AuditExportRequest) -> List[Dict[str, Any]]:
    """Simulate audit data (replace with actual data queries)"""
    
    records = []
    record_count = min(_get_estimated_record_count(data_type, request.tenant_id), 1000)  # Limit for demo
    
    for i in range(record_count):
        if data_type == AuditDataType.EXECUTION_TRACES:
            record = {
                "trace_id": str(uuid.uuid4()),
                "tenant_id": request.tenant_id,
                "workflow_id": f"workflow_{i % 10}",
                "execution_time_ms": 1000 + (i * 100),
                "success": i % 10 != 0,  # 90% success rate
                "executed_at": (datetime.utcnow() - timedelta(days=i % 30)).isoformat()
            }
        elif data_type == AuditDataType.OVERRIDE_LEDGER:
            record = {
                "override_id": str(uuid.uuid4()),
                "tenant_id": request.tenant_id,
                "override_type": "manual_approval",
                "justification": f"Override reason {i}",
                "approved_by": f"user_{i % 5}",
                "created_at": (datetime.utcnow() - timedelta(days=i % 60)).isoformat()
            }
        elif data_type == AuditDataType.TRUST_SCORES:
            record = {
                "score_id": str(uuid.uuid4()),
                "tenant_id": request.tenant_id,
                "model_id": f"model_{i % 5}",
                "trust_score": 0.7 + (i % 30) / 100,
                "computed_at": (datetime.utcnow() - timedelta(hours=i % 24)).isoformat()
            }
        else:
            record = {
                "id": str(uuid.uuid4()),
                "tenant_id": request.tenant_id,
                "data_type": data_type.value,
                "record_index": i,
                "created_at": (datetime.utcnow() - timedelta(days=i % 30)).isoformat()
            }
        
        # Apply PII anonymization if requested
        if request.anonymize_pii:
            record = _anonymize_record(record)
        
        records.append(record)
    
    return records

def _anonymize_record(record: Dict[str, Any]) -> Dict[str, Any]:
    """Anonymize PII in record"""
    
    # Simple anonymization (replace with proper anonymization logic)
    anonymized = record.copy()
    
    for key, value in record.items():
        if "email" in key.lower() or "user" in key.lower():
            if isinstance(value, str):
                anonymized[key] = f"anonymized_{hashlib.md5(value.encode()).hexdigest()[:8]}"
    
    return anonymized

def _get_data_type_description(data_type: AuditDataType) -> str:
    """Get description for data type"""
    
    descriptions = {
        AuditDataType.EXECUTION_TRACES: "Workflow execution logs and traces",
        AuditDataType.OVERRIDE_LEDGER: "Manual overrides and approvals",
        AuditDataType.EVIDENCE_PACKS: "Generated evidence and compliance packs",
        AuditDataType.POLICY_VIOLATIONS: "Policy violations and compliance issues",
        AuditDataType.TRUST_SCORES: "Trust scores and model reliability metrics",
        AuditDataType.COMPLIANCE_REPORTS: "Compliance reports and assessments",
        AuditDataType.WORKFLOW_DEFINITIONS: "Workflow definitions and configurations",
        AuditDataType.USER_ACTIVITIES: "User activities and access logs"
    }
    
    return descriptions.get(data_type, "Audit data")

def _get_estimated_record_count(data_type: AuditDataType, tenant_id: str) -> int:
    """Get estimated record count for data type"""
    
    # Simulated counts (replace with actual database queries)
    base_counts = {
        AuditDataType.EXECUTION_TRACES: 10000,
        AuditDataType.OVERRIDE_LEDGER: 500,
        AuditDataType.EVIDENCE_PACKS: 2000,
        AuditDataType.POLICY_VIOLATIONS: 100,
        AuditDataType.TRUST_SCORES: 5000,
        AuditDataType.COMPLIANCE_REPORTS: 50,
        AuditDataType.WORKFLOW_DEFINITIONS: 200,
        AuditDataType.USER_ACTIVITIES: 15000
    }
    
    return base_counts.get(data_type, 1000)

def _get_retention_period(data_type: AuditDataType) -> int:
    """Get retention period in days for data type"""
    
    retention_periods = {
        AuditDataType.EXECUTION_TRACES: 2555,  # 7 years
        AuditDataType.OVERRIDE_LEDGER: 2555,   # 7 years
        AuditDataType.EVIDENCE_PACKS: 3650,    # 10 years
        AuditDataType.POLICY_VIOLATIONS: 2555, # 7 years
        AuditDataType.TRUST_SCORES: 1095,      # 3 years
        AuditDataType.COMPLIANCE_REPORTS: 2555, # 7 years
        AuditDataType.WORKFLOW_DEFINITIONS: 1095, # 3 years
        AuditDataType.USER_ACTIVITIES: 365      # 1 year
    }
    
    return retention_periods.get(data_type, 365)

@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "service": "RBIA Open Audit APIs",
        "task": "3.4.36",
        "active_exports": len([s for s in export_status_store.values() if s.status == "processing"]),
        "completed_exports": len([s for s in export_status_store.values() if s.status == "completed"]),
        "active_api_keys": len([k for k in api_keys_store.values() if k.is_active])
    }
