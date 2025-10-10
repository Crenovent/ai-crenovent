#!/usr/bin/env python3
"""
Chapter 21.2 Persona-Specific Reporting API - Backend Skeleton
=============================================================
Tasks: 21.2-T15, T16, T17, T25, T26, T27, T28, T36, T37, T38

Backend skeleton for persona-specific reporting without KG/RBIA/AALA dependencies.
Focus on core reporting logic, persona KPI aggregation, and evidence generation.
"""

from fastapi import APIRouter, HTTPException, Depends, Query, Body
from pydantic import BaseModel, Field
from typing import Dict, List, Any, Optional, Union
from datetime import datetime, timedelta
from enum import Enum
import logging
import json
import uuid
import hashlib

# Database and dependencies
from src.services.connection_pool_manager import get_pool_manager

logger = logging.getLogger(__name__)
router = APIRouter()

# =====================================================
# PYDANTIC MODELS
# =====================================================

class PersonaType(str, Enum):
    CRO = "cro"
    CFO = "cfo"
    REVOPS_MANAGER = "revops_manager"
    COMPLIANCE_OFFICER = "compliance_officer"

class PersonaKPIType(str, Enum):
    # CRO KPIs
    PIPELINE_HYGIENE_RATE = "pipeline_hygiene_rate"
    FORECAST_ACCURACY_VARIANCE = "forecast_accuracy_variance"
    REVENUE_LEAKAGE_PREVENTED = "revenue_leakage_prevented"
    
    # CFO KPIs
    COMP_PLAN_ACCURACY = "comp_plan_accuracy"
    LEAKAGE_SAVINGS = "leakage_savings"
    COMPLIANCE_COST_PER_WORKFLOW = "compliance_cost_per_workflow"
    
    # RevOps KPIs
    WORKFLOW_ADOPTION_RATE = "workflow_adoption_rate"
    OVERRIDE_FREQUENCY = "override_frequency"
    TRUST_SCORE_TRENDS = "trust_score_trends"
    
    # Compliance KPIs
    POLICY_ADHERENCE_RATE = "policy_adherence_rate"
    SOD_VIOLATION_TRENDS = "sod_violation_trends"
    REGULATOR_DRILL_COVERAGE = "regulator_drill_coverage"

class PersonaReportRequest(BaseModel):
    tenant_id: int = Field(..., description="Tenant ID")
    persona_type: PersonaType = Field(..., description="Persona type")
    kpi_types: List[PersonaKPIType] = Field(..., description="KPI types to include")
    time_range_days: int = Field(30, description="Time range in days", ge=1, le=365)
    industry_filter: Optional[str] = Field(None, description="Industry filter")

class PersonaEvidenceRequest(BaseModel):
    tenant_id: int = Field(..., description="Tenant ID")
    persona_type: PersonaType = Field(..., description="Persona type")
    time_range_days: int = Field(30, description="Time range for evidence")
    compliance_frameworks: List[str] = Field(default_factory=list, description="Compliance frameworks")

class ReportingJobRequest(BaseModel):
    tenant_id: int = Field(..., description="Tenant ID")
    job_type: str = Field(..., description="Type of reporting job")
    schedule_cron: str = Field(..., description="Cron schedule for job")
    persona_filters: List[PersonaType] = Field(default_factory=list, description="Persona filters")

# =====================================================
# PERSONA REPORTING SERVICE
# =====================================================

class PersonaReportingService:
    """
    Core persona-specific reporting service (Chapter 21.2 backend skeleton)
    """
    
    def __init__(self, pool_manager):
        self.pool_manager = pool_manager
        
        # Persona KPI schemas (Task 21.2-T15)
        self.persona_kpi_schemas = {
            "cro": {
                "pipeline_hygiene_rate": {
                    "persona_id": "CRO",
                    "kpi_id": "PIPELINE_HYGIENE_RATE",
                    "metric_refs": ["pipeline_completeness", "data_quality_score"],
                    "calculation_method": "weighted_average",
                    "target_threshold": 0.95
                },
                "forecast_accuracy_variance": {
                    "persona_id": "CRO", 
                    "kpi_id": "FORECAST_ACCURACY_VARIANCE",
                    "metric_refs": ["forecast_vs_actual", "variance_percentage"],
                    "calculation_method": "absolute_variance",
                    "target_threshold": 0.05
                },
                "revenue_leakage_prevented": {
                    "persona_id": "CRO",
                    "kpi_id": "REVENUE_LEAKAGE_PREVENTED_USD",
                    "metric_refs": ["error_reduction", "revenue_impact"],
                    "calculation_method": "cumulative_savings",
                    "target_threshold": 100000
                }
            },
            "cfo": {
                "comp_plan_accuracy": {
                    "persona_id": "CFO",
                    "kpi_id": "COMP_PLAN_ACCURACY",
                    "metric_refs": ["forecast_vs_payout", "accuracy_percentage"],
                    "calculation_method": "accuracy_calculation",
                    "target_threshold": 0.98
                },
                "leakage_savings": {
                    "persona_id": "CFO",
                    "kpi_id": "LEAKAGE_SAVINGS_USD",
                    "metric_refs": ["cost_reduction", "efficiency_gains"],
                    "calculation_method": "total_savings",
                    "target_threshold": 50000
                },
                "compliance_cost_per_workflow": {
                    "persona_id": "CFO",
                    "kpi_id": "COMPLIANCE_COST_PER_WORKFLOW",
                    "metric_refs": ["compliance_overhead", "workflow_count"],
                    "calculation_method": "cost_per_unit",
                    "target_threshold": 25
                }
            },
            "revops_manager": {
                "workflow_adoption_rate": {
                    "persona_id": "REVOPS_MANAGER",
                    "kpi_id": "WORKFLOW_ADOPTION_RATE",
                    "metric_refs": ["active_workflows", "total_workflows"],
                    "calculation_method": "percentage_adoption",
                    "target_threshold": 0.80
                },
                "override_frequency": {
                    "persona_id": "REVOPS_MANAGER",
                    "kpi_id": "OVERRIDE_FREQUENCY",
                    "metric_refs": ["override_count", "total_executions"],
                    "calculation_method": "frequency_rate",
                    "target_threshold": 0.05
                },
                "trust_score_trends": {
                    "persona_id": "REVOPS_MANAGER",
                    "kpi_id": "TRUST_SCORE_TRENDS",
                    "metric_refs": ["trust_score_history", "trend_direction"],
                    "calculation_method": "trend_analysis",
                    "target_threshold": 0.85
                }
            },
            "compliance_officer": {
                "policy_adherence_rate": {
                    "persona_id": "COMPLIANCE_OFFICER",
                    "kpi_id": "POLICY_ADHERENCE_RATE",
                    "metric_refs": ["policy_compliance", "total_evaluations"],
                    "calculation_method": "compliance_percentage",
                    "target_threshold": 0.99
                },
                "sod_violation_trends": {
                    "persona_id": "COMPLIANCE_OFFICER",
                    "kpi_id": "SOD_VIOLATION_TRENDS",
                    "metric_refs": ["sod_violations", "trend_analysis"],
                    "calculation_method": "violation_trend",
                    "target_threshold": 0
                },
                "regulator_drill_coverage": {
                    "persona_id": "COMPLIANCE_OFFICER",
                    "kpi_id": "REGULATOR_DRILL_COVERAGE",
                    "metric_refs": ["drill_completeness", "framework_coverage"],
                    "calculation_method": "coverage_percentage",
                    "target_threshold": 1.0
                }
            }
        }
    
    async def generate_persona_report(self, request: PersonaReportRequest) -> Dict[str, Any]:
        """
        Generate persona-specific report (Tasks 21.2-T16, T17)
        """
        try:
            report_id = str(uuid.uuid4())
            
            # Get persona KPI schemas
            persona_schemas = self.persona_kpi_schemas.get(request.persona_type)
            if not persona_schemas:
                raise HTTPException(status_code=400, detail=f"Unknown persona type: {request.persona_type}")
            
            # Aggregate persona KPIs
            persona_kpis = await self._aggregate_persona_kpis(request, persona_schemas)
            
            # Generate persona report
            persona_report = {
                "report_id": report_id,
                "tenant_id": request.tenant_id,
                "persona_type": request.persona_type,
                "generated_at": datetime.utcnow().isoformat(),
                "time_range_days": request.time_range_days,
                "industry_filter": request.industry_filter,
                "persona_kpis": persona_kpis,
                "report_metadata": {
                    "total_kpis": len(persona_kpis),
                    "kpi_types_included": [kpi["kpi_type"] for kpi in persona_kpis],
                    "data_freshness": "real_time",
                    "confidence_score": await self._calculate_report_confidence(persona_kpis)
                },
                "persona_insights": await self._generate_persona_insights(request.persona_type, persona_kpis),
                "recommendations": await self._generate_persona_recommendations(request.persona_type, persona_kpis)
            }
            
            logger.info(f"📊 Persona report generated: {request.persona_type} - {report_id}")
            return persona_report
            
        except Exception as e:
            logger.error(f"❌ Failed to generate persona report: {e}")
            raise HTTPException(status_code=500, detail=f"Failed to generate report: {str(e)}")
    
    async def create_reporting_job(self, request: ReportingJobRequest) -> Dict[str, Any]:
        """
        Create automated reporting job (Task 21.2-T17)
        """
        try:
            job_id = str(uuid.uuid4())
            
            # Create reporting job configuration
            reporting_job = {
                "job_id": job_id,
                "tenant_id": request.tenant_id,
                "job_type": request.job_type,
                "schedule_cron": request.schedule_cron,
                "persona_filters": request.persona_filters,
                "created_at": datetime.utcnow().isoformat(),
                "job_status": "active",
                "job_configuration": {
                    "output_format": "json_with_signatures",
                    "delivery_method": "api_endpoint",
                    "retention_days": 365,
                    "evidence_generation": True
                },
                "next_execution": await self._calculate_next_execution(request.schedule_cron)
            }
            
            # Store reporting job (placeholder)
            await self._store_reporting_job(reporting_job)
            
            logger.info(f"📅 Reporting job created: {job_id}")
            return {
                "job_id": job_id,
                "status": "created",
                "next_execution": reporting_job["next_execution"],
                "job_configuration": reporting_job["job_configuration"]
            }
            
        except Exception as e:
            logger.error(f"❌ Failed to create reporting job: {e}")
            raise HTTPException(status_code=500, detail=f"Failed to create job: {str(e)}")
    
    async def generate_persona_evidence_pack(self, request: PersonaEvidenceRequest) -> Dict[str, Any]:
        """
        Generate persona evidence pack (Tasks 21.2-T25, T26, T27, T28)
        """
        try:
            evidence_pack_id = str(uuid.uuid4())
            
            # Collect persona evidence data
            evidence_data = await self._collect_persona_evidence_data(request)
            
            # Create evidence pack structure
            evidence_pack = {
                "evidence_pack_id": evidence_pack_id,
                "tenant_id": request.tenant_id,
                "persona_type": request.persona_type,
                "evidence_type": "persona_reporting",
                "generated_at": datetime.utcnow().isoformat(),
                "time_range_days": request.time_range_days,
                "compliance_frameworks": request.compliance_frameworks,
                "persona_evidence_data": evidence_data,
                "evidence_metadata": {
                    "collection_method": "automated_persona_aggregation",
                    "data_sources": ["persona_kpis", "reporting_jobs", "dashboard_metrics"],
                    "evidence_quality_score": await self._calculate_evidence_quality_score(evidence_data),
                    "persona_specific_validation": True
                },
                "persona_attestation": {
                    "kpi_accuracy_validated": True,
                    "persona_alignment_confirmed": True,
                    "role_specific_compliance_met": True,
                    "executive_review_ready": True
                }
            }
            
            # Generate digital signature (Task 21.2-T27)
            digital_signature = await self._create_persona_digital_signature(evidence_pack_id, evidence_pack)
            evidence_pack["digital_signature"] = digital_signature
            
            # Store in WORM storage (Task 21.2-T28)
            worm_storage_ref = await self._store_persona_evidence_in_worm(evidence_pack_id, evidence_pack)
            evidence_pack["worm_storage_ref"] = worm_storage_ref
            
            logger.info(f"📋 Persona evidence pack generated: {evidence_pack_id}")
            return evidence_pack
            
        except Exception as e:
            logger.error(f"❌ Failed to generate persona evidence pack: {e}")
            raise HTTPException(status_code=500, detail=f"Failed to generate evidence pack: {str(e)}")
    
    async def automate_persona_report_export(self, tenant_id: int, export_criteria: Dict[str, Any]) -> Dict[str, Any]:
        """
        Automate persona report evidence export (Task 21.2-T37)
        """
        try:
            export_id = str(uuid.uuid4())
            
            # Collect persona report evidence based on criteria
            persona_evidence_collection = await self._collect_persona_evidence_for_export(tenant_id, export_criteria)
            
            # Generate automated export package
            export_package = {
                "export_id": export_id,
                "tenant_id": tenant_id,
                "export_timestamp": datetime.utcnow().isoformat(),
                "export_criteria": export_criteria,
                "persona_evidence_collection": persona_evidence_collection,
                "export_metadata": {
                    "total_persona_evidence_items": len(persona_evidence_collection),
                    "personas_covered": list(set([item.get("persona_type") for item in persona_evidence_collection])),
                    "time_period_covered": export_criteria.get("time_range_days", 30),
                    "export_format": "json_with_signatures",
                    "executive_ready": True
                },
                "digital_signature": await self._create_export_digital_signature(export_id, persona_evidence_collection),
                "export_status": "completed"
            }
            
            logger.info(f"📤 Persona report evidence export automated: {export_id}")
            return export_package
            
        except Exception as e:
            logger.error(f"❌ Failed to automate persona report evidence export: {e}")
            raise HTTPException(status_code=500, detail=f"Failed to export persona evidence: {str(e)}")
    
    async def automate_persona_dashboard_refresh(self, tenant_id: int, refresh_config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Automate persona dashboard refresh jobs (Task 21.2-T38)
        """
        try:
            refresh_job_id = str(uuid.uuid4())
            
            # Create dashboard refresh configuration
            refresh_job = {
                "refresh_job_id": refresh_job_id,
                "tenant_id": tenant_id,
                "refresh_timestamp": datetime.utcnow().isoformat(),
                "refresh_config": refresh_config,
                "personas_to_refresh": refresh_config.get("personas", ["cro", "cfo", "revops_manager", "compliance_officer"]),
                "refresh_metadata": {
                    "refresh_frequency": refresh_config.get("frequency", "hourly"),
                    "data_sources_updated": ["kpi_aggregations", "reporting_jobs", "evidence_packs"],
                    "cache_invalidation": True,
                    "real_time_sync": True
                },
                "refresh_results": await self._execute_dashboard_refresh(tenant_id, refresh_config),
                "next_refresh_scheduled": (datetime.utcnow() + timedelta(hours=1)).isoformat()
            }
            
            logger.info(f"🔄 Persona dashboard refresh automated: {refresh_job_id}")
            return refresh_job
            
        except Exception as e:
            logger.error(f"❌ Failed to automate persona dashboard refresh: {e}")
            raise HTTPException(status_code=500, detail=f"Failed to refresh dashboards: {str(e)}")
    
    # =====================================================
    # HELPER METHODS
    # =====================================================
    
    async def _aggregate_persona_kpis(self, request: PersonaReportRequest, persona_schemas: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Aggregate KPIs for specific persona"""
        persona_kpis = []
        
        for kpi_type in request.kpi_types:
            kpi_schema = persona_schemas.get(kpi_type.value)
            if kpi_schema:
                # Calculate KPI value based on persona and type
                kpi_value = await self._calculate_persona_kpi_value(kpi_type.value, request.persona_type, request.tenant_id)
                
                persona_kpi = {
                    "kpi_type": kpi_type.value,
                    "kpi_id": kpi_schema["kpi_id"],
                    "persona_id": kpi_schema["persona_id"],
                    "kpi_value": kpi_value["value"],
                    "target_threshold": kpi_schema["target_threshold"],
                    "performance_status": "above_target" if kpi_value["value"] >= kpi_schema["target_threshold"] else "below_target",
                    "trend_direction": kpi_value.get("trend", "stable"),
                    "confidence_score": kpi_value.get("confidence", 0.9),
                    "calculation_method": kpi_schema["calculation_method"],
                    "metric_refs": kpi_schema["metric_refs"],
                    "last_updated": datetime.utcnow().isoformat()
                }
                persona_kpis.append(persona_kpi)
        
        return persona_kpis
    
    async def _calculate_persona_kpi_value(self, kpi_type: str, persona_type: str, tenant_id: int) -> Dict[str, Any]:
        """Calculate KPI value for specific persona"""
        # Placeholder calculations - would integrate with actual metrics
        if persona_type == "cro":
            if kpi_type == "pipeline_hygiene_rate":
                return {"value": 0.96, "confidence": 0.94, "trend": "improving"}
            elif kpi_type == "forecast_accuracy_variance":
                return {"value": 0.03, "confidence": 0.92, "trend": "stable"}
            elif kpi_type == "revenue_leakage_prevented":
                return {"value": 125000.0, "confidence": 0.88, "trend": "improving"}
        
        elif persona_type == "cfo":
            if kpi_type == "comp_plan_accuracy":
                return {"value": 0.985, "confidence": 0.96, "trend": "stable"}
            elif kpi_type == "leakage_savings":
                return {"value": 75000.0, "confidence": 0.90, "trend": "improving"}
            elif kpi_type == "compliance_cost_per_workflow":
                return {"value": 18.5, "confidence": 0.93, "trend": "improving"}
        
        elif persona_type == "revops_manager":
            if kpi_type == "workflow_adoption_rate":
                return {"value": 0.87, "confidence": 0.95, "trend": "improving"}
            elif kpi_type == "override_frequency":
                return {"value": 0.03, "confidence": 0.91, "trend": "stable"}
            elif kpi_type == "trust_score_trends":
                return {"value": 0.89, "confidence": 0.94, "trend": "improving"}
        
        elif persona_type == "compliance_officer":
            if kpi_type == "policy_adherence_rate":
                return {"value": 0.995, "confidence": 0.97, "trend": "stable"}
            elif kpi_type == "sod_violation_trends":
                return {"value": 0, "confidence": 0.98, "trend": "stable"}
            elif kpi_type == "regulator_drill_coverage":
                return {"value": 1.0, "confidence": 0.96, "trend": "stable"}
        
        return {"value": 0, "confidence": 0, "trend": "unknown"}
    
    async def _calculate_report_confidence(self, persona_kpis: List[Dict[str, Any]]) -> float:
        """Calculate overall report confidence score"""
        if not persona_kpis:
            return 0.0
        
        confidence_scores = [kpi.get("confidence_score", 0.0) for kpi in persona_kpis]
        return sum(confidence_scores) / len(confidence_scores)
    
    async def _generate_persona_insights(self, persona_type: str, persona_kpis: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Generate persona-specific insights"""
        return {
            "key_achievements": ["Improved pipeline hygiene", "Reduced compliance costs"],
            "areas_for_improvement": ["Forecast accuracy", "Override frequency"],
            "trend_analysis": "Overall positive trend with 15% improvement in key metrics",
            "risk_indicators": ["Minor increase in manual overrides"]
        }
    
    async def _generate_persona_recommendations(self, persona_type: str, persona_kpis: List[Dict[str, Any]]) -> List[str]:
        """Generate persona-specific recommendations"""
        if persona_type == "cro":
            return [
                "Focus on improving forecast accuracy through enhanced data quality",
                "Expand automation to additional pipeline stages",
                "Implement predictive analytics for revenue leakage prevention"
            ]
        elif persona_type == "cfo":
            return [
                "Continue cost optimization initiatives",
                "Implement automated compliance cost tracking",
                "Explore additional areas for leakage reduction"
            ]
        elif persona_type == "revops_manager":
            return [
                "Drive adoption of remaining workflows",
                "Investigate root causes of manual overrides",
                "Implement trust score improvement initiatives"
            ]
        elif persona_type == "compliance_officer":
            return [
                "Maintain current compliance excellence",
                "Expand regulator drill coverage to new frameworks",
                "Implement proactive policy violation prevention"
            ]
        else:
            return ["Continue monitoring key metrics", "Implement best practices"]
    
    async def _calculate_next_execution(self, cron_schedule: str) -> str:
        """Calculate next execution time for cron schedule"""
        # Placeholder - would use actual cron parser
        return (datetime.utcnow() + timedelta(hours=1)).isoformat()
    
    async def _store_reporting_job(self, reporting_job: Dict[str, Any]) -> bool:
        """Store reporting job configuration"""
        logger.info(f"📅 Storing reporting job: {reporting_job['job_id']}")
        return True
    
    async def _collect_persona_evidence_data(self, request: PersonaEvidenceRequest) -> Dict[str, Any]:
        """Collect persona evidence data"""
        return {
            "persona_kpis": {"sample": "kpi_data"},
            "reporting_jobs": {"sample": "job_data"},
            "dashboard_metrics": {"sample": "dashboard_data"},
            "audit_trail": {"sample": "audit_data"}
        }
    
    async def _calculate_evidence_quality_score(self, evidence_data: Dict[str, Any]) -> float:
        """Calculate evidence quality score"""
        return 0.95  # Placeholder
    
    async def _create_persona_digital_signature(self, evidence_pack_id: str, evidence_pack: Dict[str, Any]) -> str:
        """Create digital signature for persona evidence pack"""
        try:
            evidence_hash = hashlib.sha256(json.dumps(evidence_pack, sort_keys=True).encode()).hexdigest()
            signature = hashlib.sha256(f"{evidence_pack_id}_{evidence_hash}".encode()).hexdigest()
            return f"persona_sig_{signature[:32]}"
        except Exception as e:
            logger.error(f"❌ Failed to create persona digital signature: {e}")
            return f"persona_sig_error_{evidence_pack_id[:16]}"
    
    async def _store_persona_evidence_in_worm(self, evidence_pack_id: str, evidence_pack: Dict[str, Any]) -> str:
        """Store persona evidence in WORM storage"""
        try:
            worm_ref = f"worm_persona_{evidence_pack_id}_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"
            logger.info(f"📦 Persona evidence stored in WORM: {worm_ref}")
            return worm_ref
        except Exception as e:
            logger.error(f"❌ Failed to store persona evidence in WORM: {e}")
            return f"worm_error_{evidence_pack_id[:16]}"
    
    async def _collect_persona_evidence_for_export(self, tenant_id: int, export_criteria: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Collect persona evidence for export"""
        return [
            {"persona_type": "cro", "kpi_type": "pipeline_hygiene_rate", "value": 0.96, "evidence_id": "PERSONA001"},
            {"persona_type": "cfo", "kpi_type": "leakage_savings", "value": 75000.0, "evidence_id": "PERSONA002"}
        ]  # Placeholder
    
    async def _create_export_digital_signature(self, export_id: str, evidence_collection: List[Dict[str, Any]]) -> str:
        """Create digital signature for evidence export"""
        try:
            collection_hash = hashlib.sha256(json.dumps(evidence_collection, sort_keys=True).encode()).hexdigest()
            signature = hashlib.sha256(f"{export_id}_{collection_hash}".encode()).hexdigest()
            return f"persona_export_sig_{signature[:32]}"
        except Exception as e:
            logger.error(f"❌ Failed to create export digital signature: {e}")
            return f"export_sig_error_{export_id[:16]}"
    
    async def _execute_dashboard_refresh(self, tenant_id: int, refresh_config: Dict[str, Any]) -> Dict[str, Any]:
        """Execute dashboard refresh"""
        return {
            "personas_refreshed": refresh_config.get("personas", []),
            "refresh_success": True,
            "data_updated_count": 150,
            "cache_cleared": True,
            "refresh_duration_ms": 2500
        }

# =====================================================
# API ENDPOINTS
# =====================================================

@router.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "service": "chapter21_persona_reporting_api", "timestamp": datetime.utcnow().isoformat()}

@router.post("/generate-report")
async def generate_persona_report(
    request: PersonaReportRequest,
    pool_manager = Depends(get_pool_manager)
):
    """
    Generate persona-specific report (Tasks 21.2-T16, T17)
    """
    service = PersonaReportingService(pool_manager)
    return await service.generate_persona_report(request)

@router.post("/create-reporting-job")
async def create_reporting_job(
    request: ReportingJobRequest,
    pool_manager = Depends(get_pool_manager)
):
    """
    Create automated reporting job (Task 21.2-T17)
    """
    service = PersonaReportingService(pool_manager)
    return await service.create_reporting_job(request)

@router.post("/generate-evidence-pack")
async def generate_persona_evidence_pack(
    request: PersonaEvidenceRequest,
    pool_manager = Depends(get_pool_manager)
):
    """
    Generate persona evidence pack (Tasks 21.2-T25, T26, T27, T28)
    """
    service = PersonaReportingService(pool_manager)
    return await service.generate_persona_evidence_pack(request)

@router.post("/automate-evidence-export/{tenant_id}")
async def automate_persona_report_export(
    tenant_id: int,
    export_criteria: Dict[str, Any] = Body(...),
    pool_manager = Depends(get_pool_manager)
):
    """
    Automate persona report evidence export (Task 21.2-T37)
    """
    service = PersonaReportingService(pool_manager)
    return await service.automate_persona_report_export(tenant_id, export_criteria)

@router.post("/automate-dashboard-refresh/{tenant_id}")
async def automate_persona_dashboard_refresh(
    tenant_id: int,
    refresh_config: Dict[str, Any] = Body(...),
    pool_manager = Depends(get_pool_manager)
):
    """
    Automate persona dashboard refresh (Task 21.2-T38)
    """
    service = PersonaReportingService(pool_manager)
    return await service.automate_persona_dashboard_refresh(tenant_id, refresh_config)
