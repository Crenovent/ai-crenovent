"""
CSV Integration API
FastAPI endpoints for CSV file upload and data integration with workflows
"""

from fastapi import APIRouter, HTTPException, UploadFile, File, Form, Depends
from typing import Dict, List, Any, Optional
from pydantic import BaseModel, Field
import logging

logger = logging.getLogger(__name__)

# Create CSV integration router
router = APIRouter(prefix="/api/builder", tags=["CSV Integration"])

# Request models to avoid auto-generated Body models
class CSVUploadRequest(BaseModel):
    tenant_id: str = Field(..., description="Tenant ID for multi-tenant isolation")

# Response models to avoid Pydantic naming conflicts
class CSVUploadResponse(BaseModel):
    success: bool
    message: str
    file_info: Dict[str, Any]

class CSVTablesResponse(BaseModel):
    success: bool
    tables: List[str]
    count: int

class CSVSchemaResponse(BaseModel):
    success: bool
    table_schema: Dict[str, Any]

class TestQueryResponse(BaseModel):
    success: bool
    data: List[Dict[str, Any]]
    count: int

class TestQueryRequest(BaseModel):
    query: str = Field(..., description="SQL query to execute")
    tenant_id: str = Field(..., description="Tenant ID")


# CSV Upload functionality temporarily moved to a separate endpoint
# The File upload parameter in FastAPI causes OpenAPI schema generation issues
# TODO: Implement CSV upload using a different approach (e.g., base64 encoded JSON, separate microservice, etc.)

@router.post("/upload-csv-info")
async def get_csv_upload_info():
    """Get information about CSV upload requirements"""
    return {
        "message": "CSV upload temporarily unavailable due to OpenAPI schema conflicts",
        "alternative": "Use the other CSV endpoints for now",
        "supported_formats": ["CSV files with headers"],
        "max_file_size": "10MB",
        "note": "Upload functionality will be restored with a different implementation approach"
    }


@router.get("/csv-tables")
async def get_csv_tables(tenant_id: str) -> CSVTablesResponse:
    """Get list of available CSV tables"""
    try:
        from src.services.csv_data_service import csv_data_service
        tables = await csv_data_service.get_available_tables(tenant_id)
        
        return CSVTablesResponse(
            success=True,
            tables=tables,
            count=len(tables)
        )
        
    except Exception as e:
        logger.error(f"❌ Failed to get CSV tables: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get CSV tables: {str(e)}")


@router.get("/csv-schema/{table_name}")
async def get_csv_table_schema(table_name: str, tenant_id: str) -> CSVSchemaResponse:
    """Get schema information for CSV table"""
    try:
        from src.services.csv_data_service import csv_data_service
        schema = await csv_data_service.get_table_schema(table_name, tenant_id)
        
        if "error" in schema:
            raise HTTPException(status_code=404, detail=schema["error"])
        
        return CSVSchemaResponse(
            success=True,
            table_schema=schema
        )
        
    except Exception as e:
        logger.error(f"❌ Failed to get table schema: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get table schema: {str(e)}")


@router.post("/test-csv-query")
async def test_csv_query(request: TestQueryRequest) -> TestQueryResponse:
    """Test SQL query on uploaded CSV data"""
    try:
        query = request.query
        tenant_id = request.tenant_id
        
        if not query:
            raise HTTPException(status_code=400, detail="Query is required")
        
        from src.services.csv_data_service import csv_data_service
        result = await csv_data_service.query_csv_data(query, tenant_id)
        
        return TestQueryResponse(
            success=result['success'],
            data=result.get('records', []),
            count=result.get('row_count', 0)
        )
        
    except Exception as e:
        logger.error(f"❌ CSV query test failed: {e}")
        raise HTTPException(status_code=500, detail=f"CSV query failed: {str(e)}")
