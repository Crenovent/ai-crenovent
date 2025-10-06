"""
Data Lakehouse API
=================
Implements Chapter 11 API endpoints for Bronze/Silver/Gold data lakehouse operations

Features:
- Task 11.1.2: Set up Bronze layer (raw append-only)
- Task 11.1.3: Configure Silver layer (normalized/curated)  
- Task 11.1.4: Set up Gold layer (aggregations/features)
- Task 11.1.8: Configure Bronze ingestion pipelines (stream + batch)
- Task 11.1.9: Create Silver transformation pipelines
- Task 11.1.10: Build Gold aggregation pipelines
- Task 11.3.3: Implement data profiling jobs in Bronze
- Task 11.3.4-11.3.7: Build comprehensive data quality checks
- Task 11.4.14: Build API observability dashboards
- Task 11.4.15: Emit evidence packs per API transaction
"""

from fastapi import APIRouter, HTTPException, Depends, Body, Query
from typing import Dict, List, Any, Optional
from pydantic import BaseModel, Field
import logging
from datetime import datetime, timezone
import uuid

from dsl.knowledge.data_pipeline import (
    EnhancedDataLakehousePipeline, 
    DataLayer, 
    IngestionMode,
    TransformationStatus,
    DataQualityStatus
)
from dsl.governance.audit_trail_service import AuditTrailService, AuditEvent
from dsl.governance.evidence_pack_service import EvidencePackService
from dsl.governance.governance_hooks_middleware import GovernanceContext, GovernanceHookType, GovernancePlane

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/data-lakehouse", tags=["Data Lakehouse"])

# --- Pydantic Models for Request/Response Validation ---

class BronzeSetupRequest(BaseModel):
    tenant_id: int = Field(..., description="Tenant ID for data isolation")
    source_system: str = Field(..., description="Source system (CRM, Billing, Support, Telemetry)")
    ingestion_mode: str = Field("batch", description="Ingestion mode: stream, batch, or hybrid")
    data_format: str = Field("parquet", description="Data format: parquet, json, csv, avro")
    partition_keys: List[str] = Field(["tenant_id", "date"], description="Partition keys for data organization")
    retention_days: int = Field(90, description="Data retention in days for Bronze layer")
    region_id: str = Field("US", description="Data residency region")
    policy_pack: str = Field("default", description="Governance policy pack to apply")
    data_classification: str = Field("internal", description="Data classification level")

class SilverTransformationRequest(BaseModel):
    tenant_id: int = Field(..., description="Tenant ID for data isolation")
    bronze_path: str = Field(..., description="Source Bronze layer path")
    transformation_rules: Dict[str, Any] = Field({}, description="Transformation rules to apply")
    scd2_enabled: bool = Field(True, description="Enable SCD2 (Slowly Changing Dimensions)")
    dq_checks: List[str] = Field(["completeness", "accuracy"], description="Data quality checks to run")
    masking_rules: Dict[str, str] = Field({}, description="PII masking rules")
    retention_days: int = Field(400, description="Data retention in days for Silver layer")

class GoldAggregationRequest(BaseModel):
    tenant_id: int = Field(..., description="Tenant ID for data isolation")
    silver_paths: List[str] = Field(..., description="Source Silver layer paths")
    industry_overlay: str = Field("SaaS", description="Industry overlay: SaaS, Banking, Insurance")
    aggregation_rules: Dict[str, Any] = Field({}, description="Aggregation rules to apply")
    custom_kpis: Dict[str, Any] = Field({}, description="Custom KPI definitions")
    ml_features: Dict[str, Any] = Field({}, description="ML feature definitions")
    sla_tier: str = Field("T2", description="SLA tier: T0, T1, T2")

class DataProfilingRequest(BaseModel):
    tenant_id: int = Field(..., description="Tenant ID for data isolation")
    dataset_path: str = Field(..., description="Path to dataset for profiling")
    layer: str = Field(..., description="Data layer: bronze, silver, gold")

class DataQualityCheckRequest(BaseModel):
    tenant_id: int = Field(..., description="Tenant ID for data isolation")
    dataset_path: str = Field(..., description="Path to dataset for quality checks")
    layer: str = Field(..., description="Data layer: bronze, silver, gold")
    checks: List[str] = Field(..., description="Quality checks to run: completeness, accuracy, timeliness, consistency")

class LayerSetupResponse(BaseModel):
    success: bool
    job_id: str
    layer_path: str
    message: str
    evidence_pack_id: Optional[str] = None
    error: Optional[str] = None

class DataProfilingResponse(BaseModel):
    profile_id: str
    tenant_id: int
    dataset_path: str
    layer: str
    row_count: int
    column_count: int
    null_percentages: Dict[str, float]
    data_types: Dict[str, str]
    anomalies_detected: List[Dict[str, Any]]
    evidence_pack_id: Optional[str] = None

class DataQualityResponse(BaseModel):
    passed: bool
    check_results: List[Dict[str, Any]]
    dataset_path: str
    layer: str
    timestamp: str
    evidence_pack_id: Optional[str] = None

# Dependency to get pipeline instance
def get_data_pipeline() -> EnhancedDataLakehousePipeline:
    # In production, this would be injected via dependency injection
    return EnhancedDataLakehousePipeline()

def get_evidence_service() -> EvidencePackService:
    return EvidencePackService()

def get_audit_service() -> AuditTrailService:
    return AuditTrailService()

@router.post("/bronze/setup", response_model=LayerSetupResponse)
async def setup_bronze_layer(
    request: BronzeSetupRequest,
    pipeline: EnhancedDataLakehousePipeline = Depends(get_data_pipeline),
    evidence_service: EvidencePackService = Depends(get_evidence_service),
    audit_service: AuditTrailService = Depends(get_audit_service)
):
    """
    Task 11.1.2: Set up Bronze layer (raw append-only)
    Task 11.1.8: Configure Bronze ingestion pipelines (stream + batch)
    """
    logger.info(f"Setting up Bronze layer for tenant {request.tenant_id}")
    
    try:
        # Task 11.4.15: Emit evidence packs per API transaction
        transaction_id = str(uuid.uuid4())
        
        # Log API transaction start
        await audit_service.log_event(
            AuditEvent(
                event_id=transaction_id,
                timestamp=datetime.now(timezone.utc).isoformat(),
                service_name="DataLakehouseAPI",
                event_type="BRONZE_SETUP_INITIATED",
                tenant_id=request.tenant_id,
                user_id="api_user",
                description=f"Bronze layer setup initiated for tenant {request.tenant_id}",
                metadata={"request_data": request.dict()}
            )
        )
        
        # Configure pipeline with evidence service
        pipeline.evidence_service = evidence_service
        
        # Setup Bronze layer
        source_config = {
            "source_system": request.source_system,
            "ingestion_mode": request.ingestion_mode,
            "data_format": request.data_format,
            "partition_keys": request.partition_keys,
            "retention_days": request.retention_days,
            "region_id": request.region_id,
            "policy_pack": request.policy_pack,
            "data_classification": request.data_classification
        }
        
        result = await pipeline.setup_bronze_layer(request.tenant_id, source_config)
        
        if result["success"]:
            # Log successful completion
            await audit_service.log_event(
                AuditEvent(
                    event_id=str(uuid.uuid4()),
                    timestamp=datetime.now(timezone.utc).isoformat(),
                    service_name="DataLakehouseAPI",
                    event_type="BRONZE_SETUP_COMPLETED",
                    tenant_id=request.tenant_id,
                    user_id="api_user",
                    description=f"Bronze layer setup completed for tenant {request.tenant_id}",
                    metadata={"result": result, "transaction_id": transaction_id}
                )
            )
            
            return LayerSetupResponse(
                success=True,
                job_id=result["job_id"],
                layer_path=result["bronze_path"],
                message=result["message"],
                evidence_pack_id=result.get("evidence_pack_id")
            )
        else:
            raise HTTPException(status_code=500, detail=result.get("error", "Bronze setup failed"))
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Bronze layer setup failed for tenant {request.tenant_id}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

@router.post("/silver/setup", response_model=LayerSetupResponse)
async def setup_silver_layer(
    request: SilverTransformationRequest,
    pipeline: EnhancedDataLakehousePipeline = Depends(get_data_pipeline),
    evidence_service: EvidencePackService = Depends(get_evidence_service),
    audit_service: AuditTrailService = Depends(get_audit_service)
):
    """
    Task 11.1.3: Configure Silver layer (normalized/curated)
    Task 11.1.9: Create Silver transformation pipelines
    """
    logger.info(f"Setting up Silver layer for tenant {request.tenant_id}")
    
    try:
        transaction_id = str(uuid.uuid4())
        
        # Log API transaction start
        await audit_service.log_event(
            AuditEvent(
                event_id=transaction_id,
                timestamp=datetime.now(timezone.utc).isoformat(),
                service_name="DataLakehouseAPI",
                event_type="SILVER_SETUP_INITIATED",
                tenant_id=request.tenant_id,
                user_id="api_user",
                description=f"Silver layer setup initiated for tenant {request.tenant_id}",
                metadata={"request_data": request.dict()}
            )
        )
        
        # Configure pipeline
        pipeline.evidence_service = evidence_service
        
        # Setup Silver layer
        transformation_config = {
            "transformation_rules": request.transformation_rules,
            "scd2_enabled": request.scd2_enabled,
            "dq_checks": request.dq_checks,
            "masking_rules": request.masking_rules,
            "retention_days": request.retention_days
        }
        
        result = await pipeline.setup_silver_layer(
            request.tenant_id, 
            request.bronze_path, 
            transformation_config
        )
        
        if result["success"]:
            return LayerSetupResponse(
                success=True,
                job_id=result["job_id"],
                layer_path=result["silver_job"].target_silver_path,
                message=result["message"],
                evidence_pack_id=result.get("evidence_pack_id")
            )
        else:
            raise HTTPException(status_code=500, detail=result.get("error", "Silver setup failed"))
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Silver layer setup failed for tenant {request.tenant_id}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

@router.post("/gold/setup", response_model=LayerSetupResponse)
async def setup_gold_layer(
    request: GoldAggregationRequest,
    pipeline: EnhancedDataLakehousePipeline = Depends(get_data_pipeline),
    evidence_service: EvidencePackService = Depends(get_evidence_service),
    audit_service: AuditTrailService = Depends(get_audit_service)
):
    """
    Task 11.1.4: Set up Gold layer (aggregations/features)
    Task 11.1.10: Build Gold aggregation pipelines
    """
    logger.info(f"Setting up Gold layer for tenant {request.tenant_id}")
    
    try:
        transaction_id = str(uuid.uuid4())
        
        # Log API transaction start
        await audit_service.log_event(
            AuditEvent(
                event_id=transaction_id,
                timestamp=datetime.now(timezone.utc).isoformat(),
                service_name="DataLakehouseAPI",
                event_type="GOLD_SETUP_INITIATED",
                tenant_id=request.tenant_id,
                user_id="api_user",
                description=f"Gold layer setup initiated for tenant {request.tenant_id}",
                metadata={"request_data": request.dict()}
            )
        )
        
        # Configure pipeline
        pipeline.evidence_service = evidence_service
        
        # Setup Gold layer
        aggregation_config = {
            "industry_overlay": request.industry_overlay,
            "aggregation_rules": request.aggregation_rules,
            "custom_kpis": request.custom_kpis,
            "ml_features": request.ml_features,
            "sla_tier": request.sla_tier
        }
        
        result = await pipeline.setup_gold_layer(
            request.tenant_id, 
            request.silver_paths, 
            aggregation_config
        )
        
        if result["success"]:
            return LayerSetupResponse(
                success=True,
                job_id=result["job_id"],
                layer_path=result["gold_job"].target_gold_path,
                message=result["message"],
                evidence_pack_id=result.get("evidence_pack_id")
            )
        else:
            raise HTTPException(status_code=500, detail=result.get("error", "Gold setup failed"))
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Gold layer setup failed for tenant {request.tenant_id}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

@router.post("/profiling/run", response_model=DataProfilingResponse)
async def run_data_profiling(
    request: DataProfilingRequest,
    pipeline: EnhancedDataLakehousePipeline = Depends(get_data_pipeline),
    evidence_service: EvidencePackService = Depends(get_evidence_service)
):
    """
    Task 11.3.3: Implement data profiling jobs in Bronze
    """
    logger.info(f"Running data profiling for tenant {request.tenant_id} on {request.layer} layer")
    
    try:
        # Configure pipeline
        pipeline.evidence_service = evidence_service
        
        # Run data profiling
        layer_enum = DataLayer(request.layer.lower())
        profiling_result = await pipeline.run_data_profiling(
            request.tenant_id, 
            request.dataset_path, 
            layer_enum
        )
        
        return DataProfilingResponse(
            profile_id=profiling_result.profile_id,
            tenant_id=profiling_result.tenant_id,
            dataset_path=profiling_result.dataset_path,
            layer=profiling_result.layer.value,
            row_count=profiling_result.row_count,
            column_count=profiling_result.column_count,
            null_percentages=profiling_result.null_percentages,
            data_types=profiling_result.data_types,
            anomalies_detected=profiling_result.anomalies_detected,
            evidence_pack_id=profiling_result.evidence_pack_id
        )
        
    except Exception as e:
        logger.error(f"Data profiling failed for tenant {request.tenant_id}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

@router.post("/quality/check", response_model=DataQualityResponse)
async def run_data_quality_checks(
    request: DataQualityCheckRequest,
    pipeline: EnhancedDataLakehousePipeline = Depends(get_data_pipeline),
    evidence_service: EvidencePackService = Depends(get_evidence_service)
):
    """
    Task 11.3.4: Build completeness checks (row counts, field population)
    Task 11.3.5: Build accuracy checks (regex, reference lookups)
    Task 11.3.6: Build timeliness/freshness checks
    Task 11.3.7: Build consistency checks (Bronze→Silver→Gold)
    """
    logger.info(f"Running data quality checks for tenant {request.tenant_id} on {request.layer} layer")
    
    try:
        # Configure pipeline
        pipeline.evidence_service = evidence_service
        
        # Run data quality checks
        layer_enum = DataLayer(request.layer.lower())
        dq_results = await pipeline.run_data_quality_checks(
            request.tenant_id, 
            request.dataset_path, 
            layer_enum,
            request.checks
        )
        
        return DataQualityResponse(
            passed=dq_results["passed"],
            check_results=dq_results["check_results"],
            dataset_path=dq_results["dataset_path"],
            layer=dq_results["layer"],
            timestamp=dq_results["timestamp"],
            evidence_pack_id=dq_results.get("evidence_pack_id")
        )
        
    except Exception as e:
        logger.error(f"Data quality checks failed for tenant {request.tenant_id}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

@router.get("/layers/status/{tenant_id}")
async def get_layer_status(
    tenant_id: int,
    layer: Optional[str] = Query(None, description="Filter by layer: bronze, silver, gold")
):
    """
    Task 11.4.14: Build API observability dashboards
    Get status of data lakehouse layers for a tenant
    """
    try:
        # Simulate layer status retrieval
        layer_status = {
            "tenant_id": tenant_id,
            "layers": {
                "bronze": {
                    "status": "active",
                    "last_ingestion": "2024-12-19T10:30:00Z",
                    "row_count": 1000000,
                    "size_gb": 15.2,
                    "retention_days": 90
                },
                "silver": {
                    "status": "active", 
                    "last_transformation": "2024-12-19T11:00:00Z",
                    "row_count": 995000,
                    "size_gb": 12.8,
                    "retention_days": 400
                },
                "gold": {
                    "status": "active",
                    "last_aggregation": "2024-12-19T11:30:00Z", 
                    "kpi_count": 25,
                    "size_gb": 2.1,
                    "retention_days": 730
                }
            },
            "timestamp": datetime.now(timezone.utc).isoformat()
        }
        
        if layer:
            layer_data = layer_status["layers"].get(layer.lower())
            if not layer_data:
                raise HTTPException(status_code=404, detail=f"Layer {layer} not found")
            return {
                "tenant_id": tenant_id,
                "layer": layer.lower(),
                **layer_data,
                "timestamp": layer_status["timestamp"]
            }
        
        return layer_status
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get layer status for tenant {tenant_id}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

@router.get("/health")
async def health_check():
    """
    Task 11.4.14: Build API observability dashboards
    Health check endpoint for data lakehouse API
    """
    return {
        "status": "healthy",
        "service": "DataLakehouseAPI",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "version": "1.0.0"
    }
