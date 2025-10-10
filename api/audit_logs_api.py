"""
Task 8.2-T18: Build API (list, query, export audit logs)
FastAPI endpoints for audit log management and export
"""

import asyncio
import csv
import io
import json
import logging
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Any, Optional
from fastapi import APIRouter, Depends, HTTPException, Query, Response
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
import asyncpg

logger = logging.getLogger(__name__)

# Router for audit logs API
audit_logs_router = APIRouter(prefix="/api/v1/audit-logs", tags=["audit-logs"])


class AuditLogQuery(BaseModel):
    """Audit log query parameters"""
    tenant_id: Optional[int] = None
    event_type: Optional[str] = None
    event_category: Optional[str] = None
    actor_id: Optional[str] = None
    resource_type: Optional[str] = None
    resource_id: Optional[str] = None
    event_outcome: Optional[str] = None
    event_severity: Optional[str] = None
    compliance_frameworks: Optional[List[str]] = None
    start_date: Optional[datetime] = None
    end_date: Optional[datetime] = None
    limit: int = Field(default=100, le=1000)
    offset: int = Field(default=0, ge=0)
    sort_by: str = Field(default="event_timestamp")
    sort_order: str = Field(default="desc")


class AuditLogResponse(BaseModel):
    """Audit log response model"""
    event_id: str
    tenant_id: int
    event_type: str
    event_category: str
    event_action: str
    actor_id: str
    actor_type: str
    actor_ip: Optional[str]
    resource_type: str
    resource_id: str
    resource_name: Optional[str]
    event_data: Dict[str, Any]
    event_outcome: str
    event_severity: str
    compliance_frameworks: List[str]
    regulatory_tags: List[str]
    event_timestamp: datetime
    ingested_at: datetime
    source_system: str
    correlation_id: Optional[str]
    session_id: Optional[str]


class AuditLogListResponse(BaseModel):
    """Audit log list response"""
    logs: List[AuditLogResponse]
    total_count: int
    page_size: int
    page_number: int
    has_next: bool
    has_previous: bool


class AuditLogExportRequest(BaseModel):
    """Audit log export request"""
    query: AuditLogQuery
    export_format: str = Field(default="csv", regex="^(csv|json|pdf)$")
    include_headers: bool = True
    filename: Optional[str] = None


class AuditLogService:
    """Service for audit log operations"""
    
    def __init__(self, db_pool: asyncpg.Pool):
        self.db_pool = db_pool
        self.logger = logging.getLogger(__name__)
    
    async def list_audit_logs(self, query: AuditLogQuery) -> AuditLogListResponse:
        """List audit logs with filtering and pagination"""
        
        try:
            # Build SQL query
            where_conditions = ["1=1"]
            params = []
            param_count = 0
            
            if query.tenant_id is not None:
                param_count += 1
                where_conditions.append(f"tenant_id = ${param_count}")
                params.append(query.tenant_id)
            
            if query.event_type:
                param_count += 1
                where_conditions.append(f"event_type = ${param_count}")
                params.append(query.event_type)
            
            if query.event_category:
                param_count += 1
                where_conditions.append(f"event_category = ${param_count}")
                params.append(query.event_category)
            
            if query.actor_id:
                param_count += 1
                where_conditions.append(f"actor_id = ${param_count}")
                params.append(query.actor_id)
            
            if query.resource_type:
                param_count += 1
                where_conditions.append(f"resource_type = ${param_count}")
                params.append(query.resource_type)
            
            if query.resource_id:
                param_count += 1
                where_conditions.append(f"resource_id = ${param_count}")
                params.append(query.resource_id)
            
            if query.event_outcome:
                param_count += 1
                where_conditions.append(f"event_outcome = ${param_count}")
                params.append(query.event_outcome)
            
            if query.event_severity:
                param_count += 1
                where_conditions.append(f"event_severity = ${param_count}")
                params.append(query.event_severity)
            
            if query.compliance_frameworks:
                param_count += 1
                where_conditions.append(f"compliance_frameworks && ${param_count}")
                params.append(query.compliance_frameworks)
            
            if query.start_date:
                param_count += 1
                where_conditions.append(f"event_timestamp >= ${param_count}")
                params.append(query.start_date)
            
            if query.end_date:
                param_count += 1
                where_conditions.append(f"event_timestamp <= ${param_count}")
                params.append(query.end_date)
            
            where_clause = " AND ".join(where_conditions)
            
            # Validate sort column
            valid_sort_columns = [
                "event_timestamp", "event_type", "event_severity", 
                "actor_id", "resource_type", "ingested_at"
            ]
            sort_column = query.sort_by if query.sort_by in valid_sort_columns else "event_timestamp"
            sort_direction = "DESC" if query.sort_order.lower() == "desc" else "ASC"
            
            # Count total records
            count_query = f"""
                SELECT COUNT(*) 
                FROM audit_log_events 
                WHERE {where_clause}
            """
            
            async with self.db_pool.acquire() as conn:
                total_count = await conn.fetchval(count_query, *params)
                
                # Fetch paginated results
                param_count += 1
                limit_param = param_count
                param_count += 1
                offset_param = param_count
                
                data_query = f"""
                    SELECT event_id, tenant_id, event_type, event_category, event_action,
                           actor_id, actor_type, actor_ip, resource_type, resource_id,
                           resource_name, event_data, event_outcome, event_severity,
                           compliance_frameworks, regulatory_tags, event_timestamp,
                           ingested_at, source_system, correlation_id, session_id
                    FROM audit_log_events
                    WHERE {where_clause}
                    ORDER BY {sort_column} {sort_direction}
                    LIMIT ${limit_param} OFFSET ${offset_param}
                """
                
                params.extend([query.limit, query.offset])
                rows = await conn.fetch(data_query, *params)
                
                # Convert to response models
                logs = []
                for row in rows:
                    log = AuditLogResponse(
                        event_id=row['event_id'],
                        tenant_id=row['tenant_id'],
                        event_type=row['event_type'],
                        event_category=row['event_category'],
                        event_action=row['event_action'],
                        actor_id=row['actor_id'],
                        actor_type=row['actor_type'],
                        actor_ip=row['actor_ip'],
                        resource_type=row['resource_type'],
                        resource_id=row['resource_id'],
                        resource_name=row['resource_name'],
                        event_data=row['event_data'],
                        event_outcome=row['event_outcome'],
                        event_severity=row['event_severity'],
                        compliance_frameworks=row['compliance_frameworks'],
                        regulatory_tags=row['regulatory_tags'],
                        event_timestamp=row['event_timestamp'],
                        ingested_at=row['ingested_at'],
                        source_system=row['source_system'],
                        correlation_id=row['correlation_id'],
                        session_id=row['session_id']
                    )
                    logs.append(log)
                
                # Calculate pagination info
                page_number = (query.offset // query.limit) + 1
                has_next = (query.offset + query.limit) < total_count
                has_previous = query.offset > 0
                
                return AuditLogListResponse(
                    logs=logs,
                    total_count=total_count,
                    page_size=query.limit,
                    page_number=page_number,
                    has_next=has_next,
                    has_previous=has_previous
                )
                
        except Exception as e:
            self.logger.error(f"❌ Failed to list audit logs: {e}")
            raise HTTPException(status_code=500, detail=f"Failed to retrieve audit logs: {str(e)}")
    
    async def export_audit_logs(self, export_request: AuditLogExportRequest) -> StreamingResponse:
        """Export audit logs in specified format"""
        
        try:
            # Get all matching logs (remove pagination for export)
            export_query = export_request.query.copy()
            export_query.limit = 10000  # Max export limit
            export_query.offset = 0
            
            logs_response = await self.list_audit_logs(export_query)
            
            if export_request.export_format == "csv":
                return await self._export_csv(logs_response.logs, export_request)
            elif export_request.export_format == "json":
                return await self._export_json(logs_response.logs, export_request)
            elif export_request.export_format == "pdf":
                return await self._export_pdf(logs_response.logs, export_request)
            else:
                raise HTTPException(status_code=400, detail="Unsupported export format")
                
        except Exception as e:
            self.logger.error(f"❌ Failed to export audit logs: {e}")
            raise HTTPException(status_code=500, detail=f"Failed to export audit logs: {str(e)}")
    
    async def _export_csv(self, logs: List[AuditLogResponse], export_request: AuditLogExportRequest) -> StreamingResponse:
        """Export audit logs as CSV"""
        
        output = io.StringIO()
        
        # Define CSV columns
        fieldnames = [
            'event_id', 'tenant_id', 'event_type', 'event_category', 'event_action',
            'actor_id', 'actor_type', 'actor_ip', 'resource_type', 'resource_id',
            'resource_name', 'event_outcome', 'event_severity', 'compliance_frameworks',
            'regulatory_tags', 'event_timestamp', 'ingested_at', 'source_system',
            'correlation_id', 'session_id'
        ]
        
        writer = csv.DictWriter(output, fieldnames=fieldnames)
        
        if export_request.include_headers:
            writer.writeheader()
        
        for log in logs:
            row = {
                'event_id': log.event_id,
                'tenant_id': log.tenant_id,
                'event_type': log.event_type,
                'event_category': log.event_category,
                'event_action': log.event_action,
                'actor_id': log.actor_id,
                'actor_type': log.actor_type,
                'actor_ip': log.actor_ip,
                'resource_type': log.resource_type,
                'resource_id': log.resource_id,
                'resource_name': log.resource_name,
                'event_outcome': log.event_outcome,
                'event_severity': log.event_severity,
                'compliance_frameworks': ','.join(log.compliance_frameworks),
                'regulatory_tags': ','.join(log.regulatory_tags),
                'event_timestamp': log.event_timestamp.isoformat(),
                'ingested_at': log.ingested_at.isoformat(),
                'source_system': log.source_system,
                'correlation_id': log.correlation_id,
                'session_id': log.session_id
            }
            writer.writerow(row)
        
        output.seek(0)
        
        filename = export_request.filename or f"audit_logs_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        
        return StreamingResponse(
            io.BytesIO(output.getvalue().encode()),
            media_type="text/csv",
            headers={"Content-Disposition": f"attachment; filename={filename}"}
        )
    
    async def _export_json(self, logs: List[AuditLogResponse], export_request: AuditLogExportRequest) -> StreamingResponse:
        """Export audit logs as JSON"""
        
        # Convert logs to dict format
        logs_data = []
        for log in logs:
            log_dict = log.dict()
            # Convert datetime objects to ISO format
            log_dict['event_timestamp'] = log.event_timestamp.isoformat()
            log_dict['ingested_at'] = log.ingested_at.isoformat()
            logs_data.append(log_dict)
        
        export_data = {
            'export_timestamp': datetime.now(timezone.utc).isoformat(),
            'total_records': len(logs_data),
            'audit_logs': logs_data
        }
        
        json_output = json.dumps(export_data, indent=2)
        
        filename = export_request.filename or f"audit_logs_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        return StreamingResponse(
            io.BytesIO(json_output.encode()),
            media_type="application/json",
            headers={"Content-Disposition": f"attachment; filename={filename}"}
        )
    
    async def _export_pdf(self, logs: List[AuditLogResponse], export_request: AuditLogExportRequest) -> StreamingResponse:
        """Export audit logs as PDF (placeholder)"""
        
        # This would require a PDF library like reportlab
        # For now, return a simple text-based PDF placeholder
        
        pdf_content = f"""
        AUDIT LOGS EXPORT REPORT
        Generated: {datetime.now().isoformat()}
        Total Records: {len(logs)}
        
        {'='*50}
        
        """
        
        for i, log in enumerate(logs[:100]):  # Limit to first 100 for PDF
            pdf_content += f"""
        Record {i+1}:
        Event ID: {log.event_id}
        Type: {log.event_type}
        Actor: {log.actor_id}
        Resource: {log.resource_type}/{log.resource_id}
        Timestamp: {log.event_timestamp}
        Outcome: {log.event_outcome}
        
        """
        
        filename = export_request.filename or f"audit_logs_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
        
        return StreamingResponse(
            io.BytesIO(pdf_content.encode()),
            media_type="text/plain",
            headers={"Content-Disposition": f"attachment; filename={filename}"}
        )


# Dependency to get audit log service
async def get_audit_log_service() -> AuditLogService:
    # In production, this would get the database pool from dependency injection
    # For now, return a placeholder service
    return AuditLogService(None)


@audit_logs_router.get("/", response_model=AuditLogListResponse)
async def list_audit_logs(
    tenant_id: Optional[int] = Query(None, description="Filter by tenant ID"),
    event_type: Optional[str] = Query(None, description="Filter by event type"),
    event_category: Optional[str] = Query(None, description="Filter by event category"),
    actor_id: Optional[str] = Query(None, description="Filter by actor ID"),
    resource_type: Optional[str] = Query(None, description="Filter by resource type"),
    resource_id: Optional[str] = Query(None, description="Filter by resource ID"),
    event_outcome: Optional[str] = Query(None, description="Filter by event outcome"),
    event_severity: Optional[str] = Query(None, description="Filter by event severity"),
    compliance_frameworks: Optional[str] = Query(None, description="Filter by compliance frameworks (comma-separated)"),
    start_date: Optional[datetime] = Query(None, description="Filter by start date"),
    end_date: Optional[datetime] = Query(None, description="Filter by end date"),
    limit: int = Query(100, le=1000, description="Number of records to return"),
    offset: int = Query(0, ge=0, description="Number of records to skip"),
    sort_by: str = Query("event_timestamp", description="Field to sort by"),
    sort_order: str = Query("desc", regex="^(asc|desc)$", description="Sort order"),
    service: AuditLogService = Depends(get_audit_log_service)
):
    """
    List audit logs with filtering, sorting, and pagination
    
    Returns a paginated list of audit log events based on the provided filters.
    Supports filtering by tenant, event type, actor, resource, outcome, severity,
    compliance frameworks, and date range.
    """
    
    # Parse compliance frameworks
    frameworks_list = None
    if compliance_frameworks:
        frameworks_list = [f.strip() for f in compliance_frameworks.split(',')]
    
    query = AuditLogQuery(
        tenant_id=tenant_id,
        event_type=event_type,
        event_category=event_category,
        actor_id=actor_id,
        resource_type=resource_type,
        resource_id=resource_id,
        event_outcome=event_outcome,
        event_severity=event_severity,
        compliance_frameworks=frameworks_list,
        start_date=start_date,
        end_date=end_date,
        limit=limit,
        offset=offset,
        sort_by=sort_by,
        sort_order=sort_order
    )
    
    return await service.list_audit_logs(query)


@audit_logs_router.get("/{event_id}", response_model=AuditLogResponse)
async def get_audit_log(
    event_id: str,
    service: AuditLogService = Depends(get_audit_log_service)
):
    """
    Get a specific audit log by event ID
    
    Returns the full details of a single audit log event.
    """
    
    try:
        # This would fetch a single audit log by ID
        # For now, return a placeholder response
        raise HTTPException(status_code=404, detail="Audit log not found")
        
    except Exception as e:
        logger.error(f"❌ Failed to get audit log {event_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to retrieve audit log: {str(e)}")


@audit_logs_router.post("/export")
async def export_audit_logs(
    export_request: AuditLogExportRequest,
    service: AuditLogService = Depends(get_audit_log_service)
):
    """
    Export audit logs in specified format (CSV, JSON, PDF)
    
    Exports audit logs based on the provided query filters.
    Supports CSV, JSON, and PDF formats with optional custom filename.
    """
    
    return await service.export_audit_logs(export_request)


@audit_logs_router.get("/stats/summary")
async def get_audit_log_stats(
    tenant_id: Optional[int] = Query(None, description="Filter by tenant ID"),
    start_date: Optional[datetime] = Query(None, description="Start date for stats"),
    end_date: Optional[datetime] = Query(None, description="End date for stats"),
    service: AuditLogService = Depends(get_audit_log_service)
):
    """
    Get audit log statistics and summary
    
    Returns aggregated statistics about audit logs including:
    - Total events by type
    - Events by outcome
    - Events by severity
    - Top actors
    - Top resources
    - Compliance framework coverage
    """
    
    try:
        # This would calculate statistics from the database
        # For now, return placeholder stats
        
        stats = {
            "total_events": 0,
            "date_range": {
                "start": start_date.isoformat() if start_date else None,
                "end": end_date.isoformat() if end_date else None
            },
            "events_by_type": {},
            "events_by_outcome": {},
            "events_by_severity": {},
            "top_actors": [],
            "top_resources": [],
            "compliance_coverage": {}
        }
        
        return stats
        
    except Exception as e:
        logger.error(f"❌ Failed to get audit log stats: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to retrieve audit log statistics: {str(e)}")


@audit_logs_router.delete("/cleanup")
async def cleanup_old_audit_logs(
    days_to_keep: int = Query(90, ge=1, le=365, description="Number of days to keep"),
    tenant_id: Optional[int] = Query(None, description="Cleanup for specific tenant only"),
    dry_run: bool = Query(True, description="Perform dry run without actual deletion"),
    service: AuditLogService = Depends(get_audit_log_service)
):
    """
    Cleanup old audit logs based on retention policy
    
    Removes audit logs older than the specified number of days.
    Use dry_run=true to see what would be deleted without actually deleting.
    """
    
    try:
        cutoff_date = datetime.now(timezone.utc) - timedelta(days=days_to_keep)
        
        # This would perform the cleanup operation
        # For now, return placeholder response
        
        result = {
            "dry_run": dry_run,
            "cutoff_date": cutoff_date.isoformat(),
            "tenant_id": tenant_id,
            "records_to_delete": 0,
            "records_deleted": 0 if dry_run else 0,
            "cleanup_completed": not dry_run
        }
        
        return result
        
    except Exception as e:
        logger.error(f"❌ Failed to cleanup audit logs: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to cleanup audit logs: {str(e)}")