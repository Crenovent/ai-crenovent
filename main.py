"""
RevAI Pro - Pipeline Agents FastAPI Application
==============================================

Main FastAPI application for Policy-Aware Pipeline Agents with DSL Compiler & Runtime Engine.
Implements Chapter 6.2 (DSL Compiler), Chapter 12.1 (Builder APIs), and Pipeline Policy Integration.
"""

import asyncio
import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
import uvicorn
from datetime import datetime
from typing import Dict, Any

# Import our services
from src.services.connection_pool_manager import pool_manager
from dsl.registry.enhanced_capability_registry import EnhancedCapabilityRegistry
from dsl.hub.routing_orchestrator import RoutingOrchestrator
from dsl.knowledge.kg_store import KnowledgeGraphStore
from dsl.compiler.runtime import WorkflowRuntime

# Import governance services (Chapter 9.4 & 16)
from dsl.governance.multi_tenant_enforcer import get_multi_tenant_enforcer
from dsl.governance.policy_engine import get_policy_engine

# Import enhanced components for Tasks 14.1.x, 15.x, 16.1.x
from dsl.hub.execution_hub import ExecutionHub

# Import API routers
try:
    from src.api.builder.workflows import router as workflow_router
    from src.api.builder.templates import router as template_router  
    from src.api.builder.execution import router as execution_router
    from api.parameter_discovery import router as parameter_router
    from api.ai_builder_api import router as ai_builder_router
    from api.cross_module_api import router as cross_module_router
    from api.approval_workflow_api import router as approval_workflow_router
    from api.trust_scoring_api import router as trust_scoring_router
    from api.evidence_pack_api import router as evidence_pack_router
    from api.audit_pack_api import router as audit_pack_router
    from api.governance_dashboard_api import router as governance_dashboard_router
    from api.industry_templates_api import router as industry_templates_router
    from api.ai_approval_assistant_api import router as ai_approval_assistant_router
    from api.retry_engine_api import router as retry_engine_router
    from api.fallback_escalation_api import router as fallback_escalation_router
    from api.resilience_logging_api import router as resilience_logging_router
    from api.conflict_resolution_api import router as conflict_resolution_router
    from api.policy_pack_validation_api import router as policy_pack_validation_router
    from api.ai_policy_analyzer_api import router as ai_policy_analyzer_router
    from api.risk_simulation_api import router as risk_simulation_router
    from api.saas_adoption_api import router as saas_adoption_router
except ImportError:
    # Create fallback routers if the files don't exist
    from fastapi import APIRouter
    workflow_router = APIRouter()
    template_router = APIRouter()
    execution_router = APIRouter()
    parameter_router = APIRouter()
    ai_builder_router = APIRouter()
    cross_module_router = APIRouter()
    approval_workflow_router = APIRouter()
    trust_scoring_router = APIRouter()
    evidence_pack_router = APIRouter()
    audit_pack_router = APIRouter()
    governance_dashboard_router = APIRouter()
    industry_templates_router = APIRouter()
    ai_approval_assistant_router = APIRouter()
    retry_engine_router = APIRouter()
    fallback_escalation_router = APIRouter()
    resilience_logging_router = APIRouter()
    conflict_resolution_router = APIRouter()
    policy_pack_validation_router = APIRouter()
    ai_policy_analyzer_router = APIRouter()
    risk_simulation_router = APIRouter()
    saas_adoption_router = APIRouter()

# Import RBA Hierarchy Processor
rba_hierarchy_router = None
HIERARCHY_PROCESSOR_AVAILABLE = True  # Enable hierarchy processor for core CSV endpoints
try:
    from api.rba_hierarchy_endpoint import router as rba_hierarchy_router
    logging.info("✅ RBA Hierarchy Processor loaded successfully")
except ImportError as e:
    logging.warning(f"❌ RBA Hierarchy Processor not available: {e}")
    # Create a fallback router with basic health check
    from fastapi import APIRouter
    rba_hierarchy_router = APIRouter(prefix="/api/rba/hierarchy")
    
    @rba_hierarchy_router.get("/health")
    async def fallback_health():
        return {"status": "degraded", "message": "RBA processor unavailable"}

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifecycle management"""
    logger.info("🚀 Starting RevAI Pro Pipeline Agents...")
    
    try:
        # Initialize connection pool with error handling
        logger.info("📊 Initializing connection pool...")
        pool_success = await pool_manager.initialize()
        
        if not pool_success or pool_manager.postgres_pool is None:
            logger.error("❌ Failed to initialize database connection pool")
            logger.info("🔄 Continuing with limited functionality (no database operations)")
            # Set a flag for limited mode
            app.state.database_available = False
            app.state.pool_manager = None
            app.state.kg_store = None
            app.state.runtime = None
            app.state.registry = None
            app.state.orchestrator = None
            app.state.multi_tenant_enforcer = None
            app.state.policy_engine = None
            
            logger.info("✅ RevAI Pro Pipeline Agents initialized in LIMITED MODE!")
            logger.warning("⚠️ Database operations disabled due to connection issues")
            yield
            return
        
        # Initialize Knowledge Graph Store
        logger.info("🧠 Initializing Knowledge Graph Store...")
        kg_store = KnowledgeGraphStore(pool_manager)
        try:
            await kg_store.initialize()
        except Exception as e:
            logger.error(f"❌ Knowledge Graph initialization failed: {e}")
            logger.info("🔄 Continuing without Knowledge Graph functionality")
            kg_store = None
        
        # Initialize Workflow Runtime with atomic pipeline agents
        logger.info("⚙️ Initializing Workflow Runtime with atomic pipeline agents...")
        runtime = WorkflowRuntime(pool_manager)
        
        # Initialize Smart Capability Registry
        logger.info("🎯 Initializing Smart Capability Registry...")
        registry = EnhancedCapabilityRegistry()
        await registry.initialize()
        
        # Initialize Routing Orchestrator
        logger.info("🏛️ Initializing Policy-Aware Routing Orchestrator...")
        orchestrator = RoutingOrchestrator(pool_manager)
        await orchestrator.initialize()
        
        # Initialize Multi-Tenant Enforcer (Chapter 9.4)
        logger.info("🔒 Initializing Multi-Tenant Enforcer...")
        multi_tenant_enforcer = get_multi_tenant_enforcer(pool_manager)
        try:
            await multi_tenant_enforcer.initialize()
        except Exception as e:
            logger.error(f"❌ Multi-Tenant Enforcer initialization failed: {e}")
            logger.info("🔄 Continuing without multi-tenant enforcement")
            multi_tenant_enforcer = None
        
        # Initialize Policy Engine (Chapter 16)
        logger.info("⚖️ Initializing Policy Engine...")
        policy_engine = get_policy_engine(pool_manager)
        try:
            await policy_engine.initialize()
        except Exception as e:
            logger.error(f"❌ Policy Engine initialization failed: {e}")
            logger.info("🔄 Continuing without policy enforcement")
            policy_engine = None
        
        # Initialize Enhanced Execution Hub (Tasks 14.1.x, 15.x, 16.1.x)
        logger.info("🚀 Initializing Enhanced Execution Hub...")
        execution_hub = ExecutionHub(pool_manager)
        try:
            await execution_hub.initialize()
            logger.info("✅ Enhanced Execution Hub initialized with all SaaS intelligence components")
        except Exception as e:
            logger.error(f"❌ Enhanced Execution Hub initialization failed: {e}")
            logger.info("🔄 Continuing without enhanced execution tracking")
            execution_hub = None
        
        # Store in app state for access in endpoints
        app.state.database_available = True
        app.state.pool_manager = pool_manager
        app.state.kg_store = kg_store
        app.state.runtime = runtime
        app.state.registry = registry
        app.state.orchestrator = orchestrator
        app.state.multi_tenant_enforcer = multi_tenant_enforcer
        app.state.policy_engine = policy_engine
        app.state.execution_hub = execution_hub
        
        logger.info("✅ RevAI Pro Pipeline Agents initialized successfully!")
        logger.info("🤖 Atomic agents loaded: Data + Analysis + Action agents")
        logger.info("🏛️ Pipeline policy integration: ACTIVE")
        
        if multi_tenant_enforcer:
            logger.info("🔒 Multi-tenant enforcement: ENABLED")
        else:
            logger.warning("🔒 Multi-tenant enforcement: DISABLED (initialization failed)")
            
        if policy_engine:
            logger.info("⚖️ Compliance frameworks: SOX, GDPR, HIPAA, RBI, DPDP, NAIC")
            logger.info("📋 Evidence packs and override ledger: ACTIVE")
        else:
            logger.warning("⚖️ Compliance frameworks: DISABLED (initialization failed)")
            logger.warning("📋 Evidence packs and override ledger: DISABLED")
        yield
        
    except Exception as e:
        logger.error(f"❌ Failed to initialize application: {e}")
        raise
    finally:
        # Cleanup on shutdown
        logger.info("🔄 Shutting down RevAI Pro Pipeline Agents...")
        if hasattr(pool_manager, 'postgres_pool') and pool_manager.postgres_pool:
            await pool_manager.postgres_pool.close()

# Create FastAPI app
app = FastAPI(
    title="RevAI Pro - Pipeline Agents",
    description="Policy-Aware Pipeline Agents with Dynamic Workflow Composition & DSL Runtime Engine",
    version="2.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan
)

# CORS middleware for frontend integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include API routers
app.include_router(workflow_router, prefix="/api/builder", tags=["Workflows"])
app.include_router(template_router, prefix="/api/builder", tags=["Templates"])
app.include_router(execution_router, prefix="/api/builder", tags=["Execution"])
app.include_router(parameter_router, tags=["Parameter Discovery"])
app.include_router(ai_builder_router, tags=["AI Builder"])
app.include_router(cross_module_router, tags=["Cross-Module Orchestration"])
app.include_router(approval_workflow_router, prefix="/api/governance", tags=["Approval Workflows"])
app.include_router(trust_scoring_router, prefix="/api/trust", tags=["Trust Scoring"])
app.include_router(evidence_pack_router, prefix="/api/evidence", tags=["Evidence Packs"])
app.include_router(audit_pack_router, prefix="/api/audit", tags=["Audit Packs"])
app.include_router(governance_dashboard_router, prefix="/api/dashboards", tags=["Governance Dashboards"])
app.include_router(industry_templates_router, prefix="/api/industry", tags=["Industry Templates"])
app.include_router(ai_approval_assistant_router, prefix="/api/ai-assistant", tags=["AI Approval Assistant"])
app.include_router(retry_engine_router, prefix="/api/retry", tags=["Retry Engine"])
app.include_router(fallback_escalation_router, prefix="/api/fallback", tags=["Fallback & Escalation"])
app.include_router(resilience_logging_router, prefix="/api/resilience", tags=["Resilience Logging"])
app.include_router(conflict_resolution_router, prefix="/api/conflicts", tags=["Conflict Resolution"])
app.include_router(policy_pack_validation_router, prefix="/api/policy-validation", tags=["Policy Pack Validation"])
app.include_router(ai_policy_analyzer_router, prefix="/api/ai-policy", tags=["AI Policy Analyzer"])
app.include_router(risk_simulation_router, prefix="/api/risk-simulation", tags=["Risk Simulation"])
app.include_router(saas_adoption_router, prefix="/api/saas-adoption", tags=["SaaS Adoption"])

# Chapter 19.4 - Flywheel Activation
try:
    from api.flywheel_activation_api import router as flywheel_activation_router
    app.include_router(flywheel_activation_router, prefix="/api/flywheel", tags=["Flywheel Activation"])
except ImportError:
    logger.warning("Flywheel Activation API not found, skipping...")

try:
    from api.chapter20_dashboard_api import router as chapter20_dashboard_router
    app.include_router(chapter20_dashboard_router, prefix="/api/chapter20", tags=["Chapter 20 Dashboards"])
except ImportError:
    logger.warning("Chapter 20 Dashboard API not found, skipping...")

# Chapter 21 - Measurement & Feedback Loops (Backend Skeleton)
try:
    from api.chapter21_business_impact_api import router as chapter21_business_impact_router
    app.include_router(chapter21_business_impact_router, prefix="/api/chapter21/business-impact", tags=["Chapter 21 Business Impact"])
except ImportError:
    logger.warning("Chapter 21 Business Impact API not found, skipping...")

try:
    from api.chapter21_persona_reporting_api import router as chapter21_persona_router
    app.include_router(chapter21_persona_router, prefix="/api/chapter21/persona-reporting", tags=["Chapter 21 Persona Reporting"])
except ImportError:
    logger.warning("Chapter 21 Persona Reporting API not found, skipping...")

try:
    from api.chapter21_feedback_skeleton_api import router as chapter21_feedback_router
    app.include_router(chapter21_feedback_router, prefix="/api/chapter21/feedback-skeleton", tags=["Chapter 21 Feedback Skeleton"])
except ImportError:
    logger.warning("Chapter 21 Feedback Skeleton API not found, skipping...")

# DSL Reconciliation API (Task 7.1.16)
try:
    from api.dsl_reconciliation_api import router as dsl_reconciliation_router
    app.include_router(dsl_reconciliation_router, prefix="/api/dsl-reconciliation", tags=["DSL Reconciliation"])
except ImportError:
    logger.warning("DSL Reconciliation API not found, skipping...")

# Block Intelligence API (Task 7.3.1)
try:
    from api.block_intelligence_api import router as block_intelligence_router
    app.include_router(block_intelligence_router, prefix="/api/block-intelligence", tags=["Block Intelligence"])
except ImportError:
    logger.warning("Block Intelligence API not found, skipping...")

# Observability Taxonomy API (Task 7.4.1)
try:
    from api.observability_taxonomy_api import router as observability_taxonomy_router
    app.include_router(observability_taxonomy_router, prefix="/api/observability-taxonomy", tags=["Observability Taxonomy"])
except ImportError:
    logger.warning("Observability Taxonomy API not found, skipping...")

    # Governance Hooks API (Chapter 8.5)
    try:
        from api.governance_hooks_api import router as governance_hooks_router
        app.include_router(governance_hooks_router, prefix="/api/governance-hooks", tags=["Governance Hooks"])
    except ImportError:
        logger.warning("Governance Hooks API not found, skipping...")

    # Schema Registry API (Task 9.2.3)
    try:
        from api.schema_registry_api import router as schema_registry_router
        app.include_router(schema_registry_router, prefix="/api/schema-registry", tags=["Schema Registry"])
    except ImportError:
        logger.warning("Schema Registry API not found, skipping...")

    # Cross-Plane Orchestration API (Task 9.3.3)
    try:
        from api.cross_plane_orchestration_api import router as cross_plane_router
        app.include_router(cross_plane_router, prefix="/api/cross-plane", tags=["Cross-Plane Orchestration"])
    except ImportError:
        logger.warning("Cross-Plane Orchestration API not found, skipping...")

    # DSL Contracts API (Task 9.3.4)
    try:
        from api.dsl_contracts_api import router as dsl_contracts_router
        app.include_router(dsl_contracts_router, prefix="/api/dsl-contracts", tags=["DSL Contracts"])
    except ImportError:
        logger.warning("DSL Contracts API not found, skipping...")

    # SLA Enforcement API (Task 9.1.11)
    try:
        from api.sla_enforcement_api import router as sla_enforcement_router
        app.include_router(sla_enforcement_router, prefix="/api/sla-enforcement", tags=["SLA Enforcement"])
    except ImportError:
        logger.warning("SLA Enforcement API not found, skipping...")

# Include RBA Hierarchy Router (always include, but may be empty fallback)
if rba_hierarchy_router:
    app.include_router(rba_hierarchy_router, tags=["RBA Hierarchy Processing"])
    if HIERARCHY_PROCESSOR_AVAILABLE:
        logging.info("✅ RBA Hierarchy endpoints registered")
    else:
        logging.warning("⚠️ RBA Hierarchy fallback router registered (limited functionality)")

# Include Pipeline Agents API (our new system)
from api.workflow_builder_api import include_workflow_builder_routes

# Include all workflow builder routes (includes Universal Parameters API)
include_workflow_builder_routes(app)

try:
    from api.csv_integration import router as csv_upload_router
    app.include_router(csv_upload_router, tags=["CSV Upload & Analysis"])
except ImportError:
    logger.warning("CSV Upload API not found, skipping...")

# Include RBA Configuration API (if it exists separately)
try:
    from api.rba_config_api import router as rba_config_router
    app.include_router(rba_config_router, tags=["RBA Configuration"])
    logger.info("✅ RBA Configuration API loaded")
except ImportError:
    logger.warning("RBA Configuration API not found, skipping...")

# Include Onboarding Workflow API
try:
    from api.onboarding_workflow_api import router as onboarding_router
    app.include_router(onboarding_router, tags=["Onboarding Workflow"])
    logger.info("✅ Onboarding Workflow API loaded")
except ImportError:
    logger.warning("Onboarding Workflow API not found, skipping...")

# Include Feedback API
try:
    from api.feedback_api import router as feedback_router
    app.include_router(feedback_router, tags=["Feedback & Learning"])
    logger.info("✅ Feedback API loaded")
except ImportError:
    logger.warning("Feedback API not found, skipping...")

# Include Hierarchy Processor API
if HIERARCHY_PROCESSOR_AVAILABLE:
    try:
        # Define models directly instead of importing from non-existent file
        from pydantic import BaseModel, Field
        from typing import Dict, List, Any, Optional
        from datetime import datetime
        
        class CSVProcessRequest(BaseModel):
            csv_data: List[Dict[str, Any]] = Field(..., description="CSV data as list of dictionaries")
            tenant_id: Optional[int] = Field(default=1300, description="Tenant ID for processing")
            uploaded_by: Optional[str] = Field(default="system", description="User who uploaded the CSV")
            processing_options: Optional[Dict[str, Any]] = Field(default_factory=dict, description="Additional processing options")

        class CSVProcessResponse(BaseModel):
            success: bool = Field(..., description="Whether processing was successful")
            processed_data: List[Dict[str, Any]] = Field(default_factory=list, description="Processed CSV data")
            processing_summary: Dict[str, Any] = Field(default_factory=dict, description="Summary of processing results")
            error_message: Optional[str] = Field(default=None, description="Error message if processing failed")
            timestamp: datetime = Field(default_factory=datetime.utcnow, description="Processing timestamp")

        class HealthResponse(BaseModel):
            status: str = Field(..., description="Service status")
            timestamp: datetime = Field(default_factory=datetime.utcnow, description="Health check timestamp")
            components: Dict[str, str] = Field(default_factory=dict, description="Component statuses")
        
        from fastapi import UploadFile, File
        import pandas as pd
        import io
        
        # Import hierarchy processor components
        from hierarchy_processor.config.loader import ConfigLoader
        from hierarchy_processor.utils.validators import DataValidator
        from hierarchy_processor.core.csv_detector import CSVDetector
        from hierarchy_processor.core.field_mapper import FieldMapper
        from hierarchy_processor.core.data_normalizer import DataNormalizer
        
        # Initialize hierarchy processor components
        config_loader = ConfigLoader()
        data_validator = DataValidator()
        csv_detector = CSVDetector()
        field_mapper = FieldMapper()
        data_normalizer = DataNormalizer()

        # Initialize Enhanced Universal Mapper for Crenovent format
        from hierarchy_processor.core.enhanced_universal_mapper import EnhancedUniversalMapper
        universal_mapper = EnhancedUniversalMapper()
        
        # Initialize LLM Fallback Processor
        try:
            from hierarchy_processor.csv_llm_processor import CSVLLMProcessor
            llm_processor = CSVLLMProcessor()
            LLM_FALLBACK_AVAILABLE = llm_processor.client is not None
            if LLM_FALLBACK_AVAILABLE:
                logger.info("✅ LLM Fallback Processor available")
            else:
                logger.warning("⚠️ LLM Fallback Processor initialized but no API key available")
        except ImportError as e:
            LLM_FALLBACK_AVAILABLE = False
            llm_processor = None
            logger.warning(f"⚠️ LLM Fallback Processor not available: {e}")
        
        # Define extract_columns_with_llm function
        def extract_columns_with_llm(df, target_columns):
            """Extract columns using LLM processor"""
            if LLM_FALLBACK_AVAILABLE and llm_processor:
                try:
                    return llm_processor.extract_columns(df, target_columns)
                except Exception as e:
                    logger.error(f"LLM extraction failed: {e}")
                    return pd.DataFrame(), target_columns
            else:
                return pd.DataFrame(), target_columns
        
        @app.post("/api/hierarchy/normalize-csv-universal", tags=["Hierarchy Processor"])
        async def normalize_csv_universal(request: CSVProcessRequest):
            """Universal CSV normalization with intelligent RBA → LLM fallback"""
            try:
                logger.info(f"🚀 Universal processing: {len(request.csv_data)} records from any HRMS")

                if not request.csv_data:
                    raise HTTPException(status_code=400, detail="CSV data is empty")

                # Convert to DataFrame
                input_df = pd.DataFrame(request.csv_data)
                
                # Step 1: Try Universal Mapper (RBA approach)
                try:
                    logger.info("🎯 Attempting RBA Universal Mapper first...")
                    crenovent_df = universal_mapper.map_any_hrms_to_crenovent(input_df)
                    
                    # **CRITICAL FIX**: Clean DataFrame to prevent JSON serialization errors
                    crenovent_df = crenovent_df.fillna('')
                    for col in crenovent_df.columns:
                        crenovent_df[col] = crenovent_df[col].astype(str).replace('nan', '').replace('None', '').replace('null', '')
                    
                    # Check if mapping was successful (has valid names and emails)
                    valid_records = 0
                    for _, row in crenovent_df.iterrows():
                        name = str(row.get('Name', '')).strip()
                        email = str(row.get('Email', '')).strip()
                        if name and name != 'nan' and email and email != 'nan' and '@' in email:
                            valid_records += 1
                    
                    success_rate = valid_records / len(crenovent_df) if len(crenovent_df) > 0 else 0
                    
                    if success_rate >= 0.8:  # 80% success rate threshold
                        logger.info(f"✅ RBA Universal Mapper succeeded with {success_rate:.1%} success rate")
                        normalized_data = crenovent_df.to_dict('records')
                        
                        return {
                            "success": True,
                            "detected_system": "universal_mapper_rba",
                            "confidence": success_rate,
                            "normalized_data": normalized_data,
                            "field_mappings": getattr(universal_mapper, '_last_field_mapping', {}),  # Include detected field mappings
                            "processing_summary": {
                                "mapping_type": "rba_universal_to_crenovent",
                                "input_columns": list(input_df.columns),
                                "output_columns": list(crenovent_df.columns),
                                "total_input_records": len(request.csv_data),
                                "total_output_records": len(normalized_data),
                                "valid_records": valid_records,
                                "success_rate": f"{success_rate:.1%}",
                                "processing_time": datetime.utcnow().isoformat()
                            },
                            "metadata": {
                                "tenant_id": request.tenant_id,
                                "uploaded_by": request.uploaded_by,
                                "mapping_approach": "rba_business_rules",
                                "target_format": "crenovent_hierarchy"
                            }
                        }
                    else:
                        logger.warning(f"⚠️ RBA Universal Mapper had low success rate ({success_rate:.1%}), trying LLM fallback...")
                        
                except Exception as rba_error:
                    logger.warning(f"⚠️ RBA Universal Mapper failed: {str(rba_error)}, trying LLM fallback...")
                
                # Step 2: Try LLM Fallback if RBA failed or had low success rate
                if LLM_FALLBACK_AVAILABLE:
                    try:
                        logger.info("🤖 Attempting LLM Fallback Processor...")
                        
                        # Define target Crenovent columns
                        target_columns = [
                            'Industry', 'Org Leader', 'Role Function', 'Business Function', 'Level',
                            'Role Title', 'Name', 'Email', 'Reporting Role Function', 
                            'Reporting Manager Name', 'Reporting Manager title', 'Reporting Email',
                            'Region', 'Area', 'District', 'Territory', 'Segment', 'Modules'
                        ]
                        
                        # Use LLM to extract and map columns
                        extracted_df, missing_columns = extract_columns_with_llm(input_df, target_columns)
                        
                        if not extracted_df.empty:
                            # Fill missing columns with defaults
                            for col in missing_columns:
                                if col == 'Industry':
                                    extracted_df[col] = 'Sass'
                                elif col == 'Org Leader':
                                    extracted_df[col] = 'Org Leader'
                                elif col == 'Role Function':
                                    extracted_df[col] = 'Sales'
                                elif col == 'Business Function':
                                    extracted_df[col] = 'BF2 America'
                                elif col == 'Level':
                                    extracted_df[col] = 'M1'
                                elif col == 'Region':
                                    extracted_df[col] = 'America'
                                elif col == 'Segment':
                                    extracted_df[col] = 'SMB'
                                elif col == 'Modules':
                                    extracted_df[col] = 'Forecasting,Planning,Pipeline'
                                else:
                                    extracted_df[col] = ''
                            
                            # Ensure column order
                            extracted_df = extracted_df.reindex(columns=target_columns, fill_value='')
                            
                            normalized_data = extracted_df.to_dict('records')
                            
                            logger.info(f"✅ LLM Fallback succeeded: {len(normalized_data)} records processed")
                            
                            return {
                                "success": True,
                                "detected_system": "llm_fallback_processor",
                                "confidence": 0.9,  # High confidence for LLM processing
                                "normalized_data": normalized_data,
                                "processing_summary": {
                                    "mapping_type": "llm_ai_to_crenovent",
                                    "input_columns": list(input_df.columns),
                                    "output_columns": target_columns,
                                    "total_input_records": len(request.csv_data),
                                    "total_output_records": len(normalized_data),
                                    "missing_columns": missing_columns,
                                    "processing_time": datetime.utcnow().isoformat()
                                },
                                "metadata": {
                                    "tenant_id": request.tenant_id,
                                    "uploaded_by": request.uploaded_by,
                                    "mapping_approach": "llm_ai_semantic_mapping",
                                    "target_format": "crenovent_hierarchy"
                                }
                            }
                        else:
                            logger.error("❌ LLM Fallback: No columns could be mapped")
                            
                    except Exception as llm_error:
                        logger.error(f"❌ LLM Fallback failed: {str(llm_error)}")
                else:
                    logger.warning("❌ LLM Fallback not available - missing dependencies")
                
                # If both RBA and LLM failed, return error
                raise HTTPException(
                    status_code=500, 
                    detail="Both RBA Universal Mapper and LLM Fallback failed to process CSV"
                )

            except HTTPException:
                raise
            except Exception as e:
                logger.error(f"❌ Universal processing completely failed: {str(e)}")
                raise HTTPException(status_code=500, detail=f"Universal processing failed: {str(e)}")

        @app.post("/api/hierarchy/normalize-csv", response_model=CSVProcessResponse, tags=["Hierarchy Processor"])
        async def normalize_csv(request: CSVProcessRequest):
            """Normalize CSV data from various HRMS/CRM systems to standard format"""
            try:
                logger.info(f"Processing CSV normalization request with {len(request.csv_data)} records")
                
                if not request.csv_data:
                    raise HTTPException(status_code=400, detail="CSV data is empty")
                
                df = pd.DataFrame(request.csv_data)
                
                # Validate CSV structure
                csv_validation = data_validator.validate_csv_structure(df)
                if not csv_validation['is_valid']:
                    raise HTTPException(
                        status_code=400, 
                        detail=f"Invalid CSV structure: {'; '.join(csv_validation['errors'])}"
                    )
                
                # Detect HRMS system
                detected_system, confidence = csv_detector.detect_csv_source(df)
                logger.info(f"Detected system: {detected_system} (confidence: {confidence:.3f})")
                
                # Map fields
                mapping_result = field_mapper.map_fields(list(df.columns), detected_system)
                mappings = mapping_result['mappings']
                mapping_metadata = mapping_result['metadata']
                
                # Validate mappings
                required_fields = ['Name', 'Email']
                mapping_validation = data_validator.validate_field_mappings(mappings, required_fields)
                
                if not mapping_validation['is_valid']:
                    if detected_system != 'generic':
                        logger.warning(f"Mapping failed for {detected_system}, trying generic")
                        mapping_result = field_mapper.map_fields(list(df.columns), 'generic')
                        mappings = mapping_result['mappings']
                        mapping_validation = data_validator.validate_field_mappings(mappings, required_fields)
                        detected_system = 'generic'
                        confidence = 0.5
                
                if not mapping_validation['is_valid']:
                    raise HTTPException(
                        status_code=400,
                        detail=f"Field mapping failed: {'; '.join(mapping_validation['errors'])}"
                    )
                
                # Normalize data
                normalization_result = data_normalizer.normalize_data(df, mappings, detected_system)
                normalized_data = normalization_result['normalized_data']
                processing_stats = normalization_result['processing_stats']
                
                # Validate normalized data
                data_validation = data_validator.validate_normalized_data(normalized_data)
                hierarchy_validation = data_validator.validate_hierarchy_data(normalized_data)
                
                # Prepare response
                processing_summary = {
                    'detected_system': detected_system,
                    'detection_confidence': confidence,
                    'total_input_records': len(request.csv_data),
                    'total_output_records': len(normalized_data),
                    'processing_time': datetime.utcnow().isoformat(),
                    'mapped_fields': mapping_metadata['mapped_fields'],
                    'unmapped_headers': mapping_metadata['unmapped_headers'],
                    'processing_errors': processing_stats.get('errors', []),
                    'processing_warnings': processing_stats.get('warnings', [])
                }
                
                validation_results = {
                    'csv_validation': csv_validation,
                    'mapping_validation': mapping_validation,
                    'data_validation': data_validation,
                    'hierarchy_validation': hierarchy_validation
                }
                
                metadata = {
                    'tenant_id': request.tenant_id,
                    'uploaded_by': request.uploaded_by,
                    'field_mappings': {k: v['source_header'] if v else None for k, v in mappings.items()},
                    'mapping_confidence': mapping_metadata['mapping_confidence'],
                    'system_config_version': config_loader.load_config(detected_system).get('metadata', {}).get('version', '1.0')
                }
                
                logger.info(f"Successfully normalized {len(normalized_data)} records from {detected_system}")
                
                return CSVProcessResponse(
                    success=True,
                    detected_system=detected_system,
                    confidence=confidence,
                    normalized_data=normalized_data,
                    processing_summary=processing_summary,
                    validation_results=validation_results,
                    metadata=metadata
                )
                
            except HTTPException:
                raise
            except Exception as e:
                logger.error(f"Error processing CSV: {str(e)}")
                raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")
        
        @app.post("/api/hierarchy/upload-csv", tags=["Hierarchy Processor"])
        async def upload_csv_file(file: UploadFile = File(...)):
            """Upload CSV file directly for processing"""
            try:
                content = await file.read()
                
                # Try to detect file encoding
                try:
                    content_str = content.decode('utf-8')
                except UnicodeDecodeError:
                    try:
                        content_str = content.decode('latin-1')
                    except UnicodeDecodeError:
                        content_str = content.decode('utf-8', errors='ignore')
                
                # Parse CSV
                df = pd.read_csv(io.StringIO(content_str))
                csv_data = df.to_dict('records')
                
                # Process using the main normalize endpoint
                request = CSVProcessRequest(csv_data=csv_data)
                return await normalize_csv(request)
                
            except Exception as e:
                logger.error(f"Error uploading CSV file: {str(e)}")
                raise HTTPException(status_code=400, detail=f"Failed to process CSV file: {str(e)}")
        
        @app.get("/api/hierarchy/systems", tags=["Hierarchy Processor"])
        async def get_available_systems():
            """Get list of available HRMS/CRM systems"""
            systems = config_loader.get_available_systems()
            system_info = {}
            
            for system in systems:
                try:
                    config = config_loader.load_config(system)
                    system_info[system] = {
                        'version': config.get('metadata', {}).get('version', '1.0'),
                        'description': config.get('metadata', {}).get('description', f'{system} configuration'),
                        'supported_fields': list(config.get('fields', {}).keys())
                    }
                except Exception as e:
                    system_info[system] = {'error': str(e)}
            
            return {
                'available_systems': systems,
                'system_details': system_info
            }
        
        @app.get("/api/hierarchy/detect-system", tags=["Hierarchy Processor"])
        async def detect_system(headers: str):
            """Detect HRMS system from CSV headers (comma-separated string)"""
            try:
                header_list = [h.strip() for h in headers.split(',') if h.strip()]
                
                if not header_list:
                    raise HTTPException(status_code=400, detail="No headers provided")
                
                # Create dummy DataFrame for detection
                dummy_data = {header: ['sample'] for header in header_list}
                df = pd.DataFrame(dummy_data)
                
                detected_system, confidence = csv_detector.detect_csv_source(df)
                detection_info = csv_detector.get_detection_info(df)
                
                return {
                    'detected_system': detected_system,
                    'confidence': confidence,
                    'detection_details': detection_info
                }
                
            except Exception as e:
                logger.error(f"Error detecting system: {str(e)}")
                raise HTTPException(status_code=400, detail=f"Detection failed: {str(e)}")
        
        logger.info("✅ Hierarchy Processor API loaded and integrated")
        
    except ImportError as e:
        logger.warning(f"Failed to load Hierarchy Processor components: {e}")
else:
    logger.warning("❌ Hierarchy Processor not available - install requirements_hierarchy.txt")

# Mount static files for UI (if directory exists)
import os
if os.path.exists("frontend_integration"):
    app.mount("/ui", StaticFiles(directory="frontend_integration"), name="ui")
else:
    logger.warning("Frontend integration directory not found, skipping static files mount")

@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "service": "RevAI Pro - Pipeline Agents",
        "version": "2.0.0",
        "status": "operational",
        "timestamp": datetime.utcnow().isoformat(),
        "features": [
            "🏛️ Policy-Aware Pipeline Agents",
            "🤖 36+ Atomic Agents (Data + Analysis + Action)",
            "🧠 Semantic Intent Parsing (LLM-powered)",
            "⚙️ DSL Compiler & Runtime Engine (Chapter 6.2)",
            "🎯 Smart Capability Registry Integration",
            "🔗 Knowledge Graph Execution Tracing",
            "🏗️ Dynamic Workflow Composition",
            "🛡️ Multi-tenant Governance & Compliance"
        ],
        "endpoints": {
            "pipeline_agents": "/api/pipeline/execute",
            "sample_prompts": "/api/pipeline/sample-prompts",
            "available_agents": "/api/pipeline/agents",
            "test_system": "/api/pipeline/test"
        }
    }

@app.get("/health")
async def health_check():
    """Enhanced health check endpoint with limited mode support"""
    try:
        # Check if we're in limited mode
        database_available = getattr(app.state, 'database_available', False)
        
        if not database_available:
            return {
                "status": "limited_mode",
                "timestamp": datetime.utcnow().isoformat(),
                "mode": "limited",
                "message": "Running in limited mode due to database connection issues",
                "components": {
                    "database": "unavailable",
                    "knowledge_graph": "unavailable", 
                    "atomic_agents": "unavailable",
                    "registry": "unavailable",
                    "orchestrator": "unavailable",
                    "multi_tenant_enforcer": "unavailable",
                    "policy_engine": "unavailable"
                }
            }
        
        # Full mode - check all components
        pool_healthy = hasattr(app.state, 'pool_manager') and app.state.pool_manager and app.state.pool_manager.postgres_pool is not None
        kg_healthy = hasattr(app.state, 'kg_store') and app.state.kg_store is not None
        runtime_healthy = hasattr(app.state, 'runtime') and app.state.runtime and len(app.state.runtime.operators) > 0
        registry_healthy = hasattr(app.state, 'registry') and app.state.registry and len(app.state.registry.capabilities) > 0
        orchestrator_healthy = hasattr(app.state, 'orchestrator') and app.state.orchestrator is not None
        mt_enforcer_healthy = hasattr(app.state, 'multi_tenant_enforcer') and app.state.multi_tenant_enforcer is not None
        policy_engine_healthy = hasattr(app.state, 'policy_engine') and app.state.policy_engine is not None
        execution_hub_healthy = hasattr(app.state, 'execution_hub') and app.state.execution_hub is not None
        
        # Overall status
        core_components = [pool_healthy, runtime_healthy, registry_healthy, orchestrator_healthy]
        all_components = core_components + [kg_healthy, mt_enforcer_healthy, policy_engine_healthy, execution_hub_healthy]
        
        if all(core_components):
            if all(all_components):
                status = "healthy"
            else:
                status = "degraded"
        else:
            status = "unhealthy"
        
        return {
            "status": status,
            "timestamp": datetime.utcnow().isoformat(),
            "mode": "full",
            "components": {
                "database": "healthy" if pool_healthy else "unhealthy",
                "knowledge_graph": "healthy" if kg_healthy else "unhealthy",
                "atomic_agents": "healthy" if runtime_healthy else "unhealthy",
                "registry": "healthy" if registry_healthy else "unhealthy", 
                "orchestrator": "healthy" if orchestrator_healthy else "unhealthy",
                "multi_tenant_enforcer": "healthy" if mt_enforcer_healthy else "unhealthy",
                "policy_engine": "healthy" if policy_engine_healthy else "unhealthy",
                "execution_hub": "healthy" if execution_hub_healthy else "unhealthy"
            },
            "system_metrics": {
                "atomic_agents_count": len(app.state.runtime.operators) if runtime_healthy else 0,
                "capabilities_count": len(app.state.registry.capabilities) if registry_healthy else 0,
                "policy_integration": "active" if orchestrator_healthy else "inactive",
                "governance": "active" if (mt_enforcer_healthy and policy_engine_healthy) else "partial" if (mt_enforcer_healthy or policy_engine_healthy) else "inactive"
            }
        }
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return JSONResponse(
            status_code=503,
            content={"status": "unhealthy", "error": str(e)}
        )

if __name__ == "__main__":
    import os
    from utils.environment import env_config
    
    # Get configuration from environment
    port = env_config.get_service_port()
    host = env_config.get_service_host()
    reload = env_config.should_reload()
    
    # Log environment configuration
    config_summary = env_config.get_config_summary()
    logger.info(f"🌍 Environment: {config_summary['environment']}")
    logger.info(f"🔗 Backend URL: {config_summary['backend_url']}")
    logger.info(f"🚀 Starting server on {host}:{port}")
    
    uvicorn.run(
        "main:app",
        host=host,
        port=port,
        reload=reload,
        log_level="info"
    )
