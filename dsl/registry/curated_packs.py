"""
Task 7.3-T34: Create Curated Packs (quick-start industries)
Pack manifests with versioned sets for adoption acceleration
"""

import asyncio
import json
import uuid
from datetime import datetime, timezone, timedelta
from enum import Enum
from typing import Dict, List, Optional, Any, Set
from dataclasses import dataclass, field
from pathlib import Path

import yaml
import asyncpg


class PackType(Enum):
    """Types of curated packs"""
    INDUSTRY_STARTER = "industry_starter"
    USE_CASE_BUNDLE = "use_case_bundle"
    COMPLIANCE_PACK = "compliance_pack"
    INTEGRATION_PACK = "integration_pack"
    BEST_PRACTICES = "best_practices"
    DEMO_SHOWCASE = "demo_showcase"


class IndustryVertical(Enum):
    """Industry verticals for packs"""
    SAAS = "SaaS"
    BANKING = "Banking"
    INSURANCE = "Insurance"
    HEALTHCARE = "Healthcare"
    ECOMMERCE = "E-commerce"
    FINTECH = "FinTech"
    MANUFACTURING = "Manufacturing"
    UNIVERSAL = "Universal"


class PackStatus(Enum):
    """Pack lifecycle status"""
    DRAFT = "draft"
    REVIEW = "review"
    PUBLISHED = "published"
    DEPRECATED = "deprecated"
    ARCHIVED = "archived"


class PackComplexity(Enum):
    """Pack complexity levels"""
    BEGINNER = "beginner"
    INTERMEDIATE = "intermediate"
    ADVANCED = "advanced"
    EXPERT = "expert"


@dataclass
class PackWorkflow:
    """Workflow reference in a pack"""
    workflow_id: str
    version_id: Optional[str] = None
    version_alias: str = "stable"
    required: bool = True
    category: str = "core"
    description: Optional[str] = None
    configuration_overrides: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "workflow_id": self.workflow_id,
            "version_id": self.version_id,
            "version_alias": self.version_alias,
            "required": self.required,
            "category": self.category,
            "description": self.description,
            "configuration_overrides": self.configuration_overrides
        }


@dataclass
class PackPolicy:
    """Policy reference in a pack"""
    policy_pack_id: str
    policy_version: Optional[str] = None
    enforcement_level: str = "required"
    scope: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "policy_pack_id": self.policy_pack_id,
            "policy_version": self.policy_version,
            "enforcement_level": self.enforcement_level,
            "scope": self.scope
        }


@dataclass
class PackDependency:
    """External dependency for a pack"""
    dependency_type: str  # connector, service, api
    name: str
    version_requirement: Optional[str] = None
    optional: bool = False
    configuration_template: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "dependency_type": self.dependency_type,
            "name": self.name,
            "version_requirement": self.version_requirement,
            "optional": self.optional,
            "configuration_template": self.configuration_template
        }


@dataclass
class CuratedPack:
    """Complete curated pack definition"""
    pack_id: str
    pack_name: str
    pack_type: PackType
    industry_vertical: IndustryVertical
    complexity: PackComplexity
    version: str
    status: PackStatus
    
    # Content
    workflows: List[PackWorkflow] = field(default_factory=list)
    policies: List[PackPolicy] = field(default_factory=list)
    dependencies: List[PackDependency] = field(default_factory=list)
    
    # Metadata
    description: str = ""
    long_description: str = ""
    tags: List[str] = field(default_factory=list)
    keywords: List[str] = field(default_factory=list)
    
    # Documentation
    readme_content: str = ""
    setup_instructions: str = ""
    examples: List[Dict[str, Any]] = field(default_factory=list)
    
    # Lifecycle
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    created_by: str = "system"
    
    # Usage tracking
    download_count: int = 0
    success_rate: float = 0.0
    user_rating: float = 0.0
    
    @property
    def required_workflows(self) -> List[PackWorkflow]:
        return [w for w in self.workflows if w.required]
    
    @property
    def optional_workflows(self) -> List[PackWorkflow]:
        return [w for w in self.workflows if not w.required]
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "pack_id": self.pack_id,
            "pack_name": self.pack_name,
            "pack_type": self.pack_type.value,
            "industry_vertical": self.industry_vertical.value,
            "complexity": self.complexity.value,
            "version": self.version,
            "status": self.status.value,
            "workflows": [w.to_dict() for w in self.workflows],
            "policies": [p.to_dict() for p in self.policies],
            "dependencies": [d.to_dict() for d in self.dependencies],
            "description": self.description,
            "long_description": self.long_description,
            "tags": self.tags,
            "keywords": self.keywords,
            "readme_content": self.readme_content,
            "setup_instructions": self.setup_instructions,
            "examples": self.examples,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "created_by": self.created_by,
            "download_count": self.download_count,
            "success_rate": self.success_rate,
            "user_rating": self.user_rating
        }


class CuratedPackManager:
    """Manager for curated workflow packs"""
    
    def __init__(self, db_pool: Optional[asyncpg.Pool] = None):
        self.db_pool = db_pool
        self.packs: Dict[str, CuratedPack] = {}
        self._initialize_default_packs()
    
    def _initialize_default_packs(self) -> None:
        """Initialize default curated packs"""
        
        # SaaS Starter Pack
        saas_starter = self._create_saas_starter_pack()
        self.packs[saas_starter.pack_id] = saas_starter
        
        # Banking Compliance Pack
        banking_compliance = self._create_banking_compliance_pack()
        self.packs[banking_compliance.pack_id] = banking_compliance
        
        # Insurance Regulatory Pack
        insurance_regulatory = self._create_insurance_regulatory_pack()
        self.packs[insurance_regulatory.pack_id] = insurance_regulatory
        
        # Revenue Operations Best Practices
        revops_best_practices = self._create_revops_best_practices_pack()
        self.packs[revops_best_practices.pack_id] = revops_best_practices
        
        # Demo Showcase Pack
        demo_showcase = self._create_demo_showcase_pack()
        self.packs[demo_showcase.pack_id] = demo_showcase
    
    def _create_saas_starter_pack(self) -> CuratedPack:
        """Create SaaS industry starter pack"""
        
        pack = CuratedPack(
            pack_id="saas_starter_v1",
            pack_name="SaaS Revenue Operations Starter Pack",
            pack_type=PackType.INDUSTRY_STARTER,
            industry_vertical=IndustryVertical.SAAS,
            complexity=PackComplexity.BEGINNER,
            version="1.0.0",
            status=PackStatus.PUBLISHED,
            description="Essential workflows for SaaS revenue operations teams",
            long_description="""
            This starter pack provides the fundamental workflows needed to get started with 
            revenue operations automation in a SaaS environment. Includes pipeline hygiene, 
            forecast accuracy, lead scoring, and basic compliance workflows.
            
            Perfect for teams new to RevOps automation or migrating from manual processes.
            """,
            tags=["saas", "starter", "revops", "pipeline", "forecasting"],
            keywords=["revenue", "operations", "automation", "saas", "crm"],
            readme_content=self._get_saas_starter_readme(),
            setup_instructions=self._get_saas_starter_setup(),
            created_by="RevAI Team"
        )
        
        # Core workflows
        pack.workflows = [
            PackWorkflow(
                workflow_id="pipeline_hygiene_basic",
                version_alias="stable",
                required=True,
                category="core",
                description="Basic pipeline data quality checks"
            ),
            PackWorkflow(
                workflow_id="lead_scoring_standard",
                version_alias="stable",
                required=True,
                category="core",
                description="Standard lead scoring algorithm"
            ),
            PackWorkflow(
                workflow_id="forecast_accuracy_tracker",
                version_alias="stable",
                required=True,
                category="core",
                description="Track and improve forecast accuracy"
            ),
            PackWorkflow(
                workflow_id="opportunity_stage_automation",
                version_alias="stable",
                required=False,
                category="enhancement",
                description="Automate opportunity stage progression"
            ),
            PackWorkflow(
                workflow_id="customer_health_scoring",
                version_alias="stable",
                required=False,
                category="enhancement",
                description="Monitor customer health metrics"
            )
        ]
        
        # Policies
        pack.policies = [
            PackPolicy(
                policy_pack_id="saas_data_governance",
                enforcement_level="required",
                scope=["data_quality", "privacy"]
            ),
            PackPolicy(
                policy_pack_id="gdpr_compliance_basic",
                enforcement_level="recommended",
                scope=["privacy", "data_retention"]
            )
        ]
        
        # Dependencies
        pack.dependencies = [
            PackDependency(
                dependency_type="connector",
                name="salesforce_connector",
                version_requirement=">=2.0.0",
                optional=False,
                configuration_template={
                    "api_version": "v52.0",
                    "sandbox_mode": False,
                    "bulk_api_enabled": True
                }
            ),
            PackDependency(
                dependency_type="service",
                name="email_service",
                optional=True,
                configuration_template={
                    "provider": "sendgrid",
                    "template_support": True
                }
            )
        ]
        
        # Examples
        pack.examples = [
            {
                "name": "Basic Pipeline Setup",
                "description": "Set up basic pipeline hygiene workflow",
                "configuration": {
                    "pipeline_stages": ["Lead", "Qualified", "Proposal", "Closed Won", "Closed Lost"],
                    "required_fields": ["amount", "close_date", "stage"],
                    "validation_rules": {
                        "amount_minimum": 100,
                        "close_date_future": True
                    }
                }
            }
        ]
        
        return pack
    
    def _create_banking_compliance_pack(self) -> CuratedPack:
        """Create banking compliance pack"""
        
        pack = CuratedPack(
            pack_id="banking_compliance_v1",
            pack_name="Banking Regulatory Compliance Pack",
            pack_type=PackType.COMPLIANCE_PACK,
            industry_vertical=IndustryVertical.BANKING,
            complexity=PackComplexity.ADVANCED,
            version="1.0.0",
            status=PackStatus.PUBLISHED,
            description="Comprehensive compliance workflows for banking institutions",
            long_description="""
            Regulatory compliance workflows designed for banking and financial institutions.
            Includes RBI compliance, KYC automation, AML monitoring, and audit trail generation.
            
            Designed for regulated financial institutions requiring strict compliance controls.
            """,
            tags=["banking", "compliance", "rbi", "kyc", "aml"],
            keywords=["banking", "compliance", "regulatory", "kyc", "aml", "rbi"],
            created_by="Compliance Team"
        )
        
        # Compliance workflows
        pack.workflows = [
            PackWorkflow(
                workflow_id="kyc_verification_automated",
                version_alias="stable",
                required=True,
                category="compliance",
                description="Automated KYC verification process"
            ),
            PackWorkflow(
                workflow_id="aml_monitoring_realtime",
                version_alias="stable",
                required=True,
                category="compliance",
                description="Real-time AML transaction monitoring"
            ),
            PackWorkflow(
                workflow_id="rbi_reporting_automated",
                version_alias="stable",
                required=True,
                category="reporting",
                description="Automated RBI regulatory reporting"
            ),
            PackWorkflow(
                workflow_id="loan_classification_npa",
                version_alias="stable",
                required=False,
                category="risk",
                description="NPA loan classification automation"
            )
        ]
        
        # Strict compliance policies
        pack.policies = [
            PackPolicy(
                policy_pack_id="rbi_compliance_strict",
                enforcement_level="required",
                scope=["data_localization", "audit_trail", "access_control"]
            ),
            PackPolicy(
                policy_pack_id="basel_iii_compliance",
                enforcement_level="required",
                scope=["risk_management", "capital_adequacy"]
            )
        ]
        
        return pack
    
    def _create_insurance_regulatory_pack(self) -> CuratedPack:
        """Create insurance regulatory pack"""
        
        pack = CuratedPack(
            pack_id="insurance_regulatory_v1",
            pack_name="Insurance Regulatory Compliance Pack",
            pack_type=PackType.COMPLIANCE_PACK,
            industry_vertical=IndustryVertical.INSURANCE,
            complexity=PackComplexity.ADVANCED,
            version="1.0.0",
            status=PackStatus.PUBLISHED,
            description="IRDAI compliance and insurance operations workflows",
            long_description="""
            Comprehensive insurance regulatory compliance workflows for IRDAI requirements.
            Includes solvency monitoring, claims processing automation, and regulatory reporting.
            """,
            tags=["insurance", "irdai", "solvency", "claims", "regulatory"],
            keywords=["insurance", "regulatory", "irdai", "solvency", "claims"],
            created_by="Insurance Team"
        )
        
        pack.workflows = [
            PackWorkflow(
                workflow_id="solvency_monitoring_realtime",
                version_alias="stable",
                required=True,
                category="compliance",
                description="Real-time solvency ratio monitoring"
            ),
            PackWorkflow(
                workflow_id="claims_processing_automated",
                version_alias="stable",
                required=True,
                category="operations",
                description="Automated claims processing workflow"
            ),
            PackWorkflow(
                workflow_id="irdai_reporting_quarterly",
                version_alias="stable",
                required=True,
                category="reporting",
                description="Quarterly IRDAI regulatory reporting"
            )
        ]
        
        pack.policies = [
            PackPolicy(
                policy_pack_id="irdai_compliance_full",
                enforcement_level="required",
                scope=["solvency", "claims", "reporting"]
            )
        ]
        
        return pack
    
    def _create_revops_best_practices_pack(self) -> CuratedPack:
        """Create RevOps best practices pack"""
        
        pack = CuratedPack(
            pack_id="revops_best_practices_v1",
            pack_name="Revenue Operations Best Practices Pack",
            pack_type=PackType.BEST_PRACTICES,
            industry_vertical=IndustryVertical.UNIVERSAL,
            complexity=PackComplexity.INTERMEDIATE,
            version="1.0.0",
            status=PackStatus.PUBLISHED,
            description="Industry best practices for revenue operations",
            long_description="""
            Curated collection of revenue operations best practices across industries.
            Includes advanced forecasting, territory management, compensation planning,
            and performance analytics workflows.
            """,
            tags=["revops", "best-practices", "forecasting", "territory", "compensation"],
            keywords=["revenue", "operations", "best-practices", "forecasting"],
            created_by="RevOps Community"
        )
        
        pack.workflows = [
            PackWorkflow(
                workflow_id="advanced_forecasting_ml",
                version_alias="stable",
                required=True,
                category="forecasting",
                description="ML-enhanced forecasting workflow"
            ),
            PackWorkflow(
                workflow_id="territory_optimization",
                version_alias="stable",
                required=False,
                category="planning",
                description="AI-driven territory optimization"
            ),
            PackWorkflow(
                workflow_id="compensation_planning_advanced",
                version_alias="stable",
                required=False,
                category="planning",
                description="Advanced compensation planning"
            )
        ]
        
        return pack
    
    def _create_demo_showcase_pack(self) -> CuratedPack:
        """Create demo showcase pack"""
        
        pack = CuratedPack(
            pack_id="demo_showcase_v1",
            pack_name="RevAI Pro Demo Showcase",
            pack_type=PackType.DEMO_SHOWCASE,
            industry_vertical=IndustryVertical.UNIVERSAL,
            complexity=PackComplexity.BEGINNER,
            version="1.0.0",
            status=PackStatus.PUBLISHED,
            description="Interactive demo workflows showcasing RevAI Pro capabilities",
            long_description="""
            Interactive demonstration workflows that showcase the full capabilities of RevAI Pro.
            Perfect for evaluations, training, and proof-of-concept implementations.
            """,
            tags=["demo", "showcase", "training", "poc"],
            keywords=["demo", "showcase", "training", "evaluation"],
            created_by="Demo Team"
        )
        
        pack.workflows = [
            PackWorkflow(
                workflow_id="demo_pipeline_hygiene",
                version_alias="stable",
                required=True,
                category="demo",
                description="Demo: Pipeline hygiene automation"
            ),
            PackWorkflow(
                workflow_id="demo_lead_scoring",
                version_alias="stable",
                required=True,
                category="demo",
                description="Demo: AI-powered lead scoring"
            ),
            PackWorkflow(
                workflow_id="demo_forecast_accuracy",
                version_alias="stable",
                required=True,
                category="demo",
                description="Demo: Forecast accuracy tracking"
            )
        ]
        
        return pack
    
    def _get_saas_starter_readme(self) -> str:
        """Get README content for SaaS starter pack"""
        return """
# SaaS Revenue Operations Starter Pack

## Overview
This pack provides essential workflows for SaaS revenue operations teams getting started with automation.

## Included Workflows
- **Pipeline Hygiene**: Automated data quality checks
- **Lead Scoring**: Standard lead scoring algorithm  
- **Forecast Accuracy**: Track and improve forecast precision
- **Opportunity Stage Automation**: Streamline stage progression
- **Customer Health Scoring**: Monitor customer success metrics

## Prerequisites
- Salesforce CRM integration
- Basic RevOps team training
- Data governance policies in place

## Quick Start
1. Install the pack using the RevAI Pro interface
2. Configure Salesforce connector credentials
3. Review and customize workflow parameters
4. Enable workflows in your tenant
5. Monitor execution through dashboards

## Support
For questions and support, contact the RevAI Pro team.
        """.strip()
    
    def _get_saas_starter_setup(self) -> str:
        """Get setup instructions for SaaS starter pack"""
        return """
## Setup Instructions

### Step 1: Prerequisites
- Ensure Salesforce integration is configured
- Verify user permissions for RevOps workflows
- Review data governance policies

### Step 2: Installation
1. Navigate to Pack Marketplace in RevAI Pro
2. Select "SaaS Revenue Operations Starter Pack"
3. Click "Install Pack"
4. Review workflow permissions and approve

### Step 3: Configuration
1. Configure Salesforce connector settings
2. Set pipeline stage mappings
3. Define lead scoring criteria
4. Customize forecast parameters

### Step 4: Testing
1. Run workflows in test mode
2. Validate data quality checks
3. Review generated evidence packs
4. Confirm compliance with policies

### Step 5: Production Deployment
1. Enable workflows for production use
2. Set up monitoring dashboards
3. Configure alert notifications
4. Train team on new processes
        """.strip()
    
    async def get_pack(self, pack_id: str) -> Optional[CuratedPack]:
        """Get a curated pack by ID"""
        
        # Check in-memory cache first
        if pack_id in self.packs:
            return self.packs[pack_id]
        
        # Check database if available
        if self.db_pool:
            try:
                async with self.db_pool.acquire() as conn:
                    row = await conn.fetchrow(
                        "SELECT * FROM curated_packs WHERE pack_id = $1",
                        pack_id
                    )
                    
                    if row:
                        pack_data = json.loads(row["pack_data"])
                        return self._deserialize_pack(pack_data)
            except Exception:
                pass
        
        return None
    
    async def list_packs(
        self,
        industry: Optional[IndustryVertical] = None,
        pack_type: Optional[PackType] = None,
        complexity: Optional[PackComplexity] = None,
        status: PackStatus = PackStatus.PUBLISHED
    ) -> List[CuratedPack]:
        """List available curated packs with filters"""
        
        packs = []
        
        # Get from in-memory cache
        for pack in self.packs.values():
            if status and pack.status != status:
                continue
            if industry and pack.industry_vertical != industry:
                continue
            if pack_type and pack.pack_type != pack_type:
                continue
            if complexity and pack.complexity != complexity:
                continue
            
            packs.append(pack)
        
        # Get from database if available
        if self.db_pool:
            try:
                query_parts = ["SELECT * FROM curated_packs WHERE status = $1"]
                params = [status.value]
                param_count = 1
                
                if industry:
                    param_count += 1
                    query_parts.append(f"AND industry_vertical = ${param_count}")
                    params.append(industry.value)
                
                if pack_type:
                    param_count += 1
                    query_parts.append(f"AND pack_type = ${param_count}")
                    params.append(pack_type.value)
                
                if complexity:
                    param_count += 1
                    query_parts.append(f"AND complexity = ${param_count}")
                    params.append(complexity.value)
                
                query = " ".join(query_parts) + " ORDER BY created_at DESC"
                
                async with self.db_pool.acquire() as conn:
                    rows = await conn.fetch(query, *params)
                    
                    for row in rows:
                        pack_data = json.loads(row["pack_data"])
                        pack = self._deserialize_pack(pack_data)
                        
                        # Avoid duplicates from in-memory cache
                        if pack.pack_id not in [p.pack_id for p in packs]:
                            packs.append(pack)
            except Exception:
                pass
        
        return packs
    
    async def install_pack(
        self,
        pack_id: str,
        tenant_id: int,
        installed_by: str,
        configuration_overrides: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Install a curated pack for a tenant"""
        
        pack = await self.get_pack(pack_id)
        if not pack:
            return {"success": False, "error": "Pack not found"}
        
        installation_id = str(uuid.uuid4())
        
        try:
            # Install workflows
            installed_workflows = []
            for workflow_ref in pack.workflows:
                if workflow_ref.required or configuration_overrides.get("install_optional", False):
                    # This would integrate with workflow registry to install workflows
                    installed_workflows.append({
                        "workflow_id": workflow_ref.workflow_id,
                        "version_alias": workflow_ref.version_alias,
                        "status": "installed"
                    })
            
            # Apply policies
            applied_policies = []
            for policy_ref in pack.policies:
                # This would integrate with policy engine to apply policies
                applied_policies.append({
                    "policy_pack_id": policy_ref.policy_pack_id,
                    "enforcement_level": policy_ref.enforcement_level,
                    "status": "applied"
                })
            
            # Record installation
            if self.db_pool:
                await self._record_pack_installation(
                    installation_id, pack_id, tenant_id, installed_by,
                    installed_workflows, applied_policies
                )
            
            # Update download count
            pack.download_count += 1
            
            return {
                "success": True,
                "installation_id": installation_id,
                "installed_workflows": installed_workflows,
                "applied_policies": applied_policies,
                "setup_instructions": pack.setup_instructions
            }
            
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    async def create_custom_pack(
        self,
        pack_name: str,
        pack_type: PackType,
        industry_vertical: IndustryVertical,
        workflow_ids: List[str],
        created_by: str,
        description: str = "",
        complexity: PackComplexity = PackComplexity.INTERMEDIATE
    ) -> CuratedPack:
        """Create a custom curated pack"""
        
        pack_id = f"custom_{uuid.uuid4().hex[:8]}"
        
        # Create workflow references
        workflows = []
        for workflow_id in workflow_ids:
            workflows.append(PackWorkflow(
                workflow_id=workflow_id,
                version_alias="stable",
                required=True,
                category="custom"
            ))
        
        pack = CuratedPack(
            pack_id=pack_id,
            pack_name=pack_name,
            pack_type=pack_type,
            industry_vertical=industry_vertical,
            complexity=complexity,
            version="1.0.0",
            status=PackStatus.DRAFT,
            workflows=workflows,
            description=description,
            created_by=created_by
        )
        
        # Store in cache
        self.packs[pack_id] = pack
        
        # Store in database if available
        if self.db_pool:
            await self._store_pack(pack)
        
        return pack
    
    async def get_pack_analytics(self, pack_id: str) -> Dict[str, Any]:
        """Get analytics for a curated pack"""
        
        pack = await self.get_pack(pack_id)
        if not pack:
            return {"error": "Pack not found"}
        
        analytics = {
            "pack_id": pack_id,
            "download_count": pack.download_count,
            "success_rate": pack.success_rate,
            "user_rating": pack.user_rating,
            "installation_trend": [],
            "usage_metrics": {}
        }
        
        if self.db_pool:
            try:
                async with self.db_pool.acquire() as conn:
                    # Get installation trend
                    trend_query = """
                        SELECT DATE_TRUNC('week', installed_at) as week,
                               COUNT(*) as installations
                        FROM pack_installations 
                        WHERE pack_id = $1 
                        AND installed_at >= NOW() - INTERVAL '12 weeks'
                        GROUP BY week 
                        ORDER BY week
                    """
                    
                    trend_rows = await conn.fetch(trend_query, pack_id)
                    analytics["installation_trend"] = [
                        {
                            "week": row["week"].isoformat(),
                            "installations": row["installations"]
                        }
                        for row in trend_rows
                    ]
                    
                    # Get usage metrics
                    usage_query = """
                        SELECT 
                            COUNT(DISTINCT tenant_id) as unique_tenants,
                            AVG(success_rate) as avg_success_rate,
                            COUNT(*) as total_installations
                        FROM pack_installations 
                        WHERE pack_id = $1
                    """
                    
                    usage_row = await conn.fetchrow(usage_query, pack_id)
                    if usage_row:
                        analytics["usage_metrics"] = {
                            "unique_tenants": usage_row["unique_tenants"] or 0,
                            "avg_success_rate": float(usage_row["avg_success_rate"] or 0),
                            "total_installations": usage_row["total_installations"] or 0
                        }
            except Exception:
                pass
        
        return analytics
    
    def _deserialize_pack(self, pack_data: Dict[str, Any]) -> CuratedPack:
        """Deserialize pack data from database"""
        
        workflows = [
            PackWorkflow(
                workflow_id=w["workflow_id"],
                version_id=w.get("version_id"),
                version_alias=w.get("version_alias", "stable"),
                required=w.get("required", True),
                category=w.get("category", "core"),
                description=w.get("description"),
                configuration_overrides=w.get("configuration_overrides", {})
            )
            for w in pack_data.get("workflows", [])
        ]
        
        policies = [
            PackPolicy(
                policy_pack_id=p["policy_pack_id"],
                policy_version=p.get("policy_version"),
                enforcement_level=p.get("enforcement_level", "required"),
                scope=p.get("scope", [])
            )
            for p in pack_data.get("policies", [])
        ]
        
        dependencies = [
            PackDependency(
                dependency_type=d["dependency_type"],
                name=d["name"],
                version_requirement=d.get("version_requirement"),
                optional=d.get("optional", False),
                configuration_template=d.get("configuration_template", {})
            )
            for d in pack_data.get("dependencies", [])
        ]
        
        return CuratedPack(
            pack_id=pack_data["pack_id"],
            pack_name=pack_data["pack_name"],
            pack_type=PackType(pack_data["pack_type"]),
            industry_vertical=IndustryVertical(pack_data["industry_vertical"]),
            complexity=PackComplexity(pack_data["complexity"]),
            version=pack_data["version"],
            status=PackStatus(pack_data["status"]),
            workflows=workflows,
            policies=policies,
            dependencies=dependencies,
            description=pack_data.get("description", ""),
            long_description=pack_data.get("long_description", ""),
            tags=pack_data.get("tags", []),
            keywords=pack_data.get("keywords", []),
            readme_content=pack_data.get("readme_content", ""),
            setup_instructions=pack_data.get("setup_instructions", ""),
            examples=pack_data.get("examples", []),
            created_at=datetime.fromisoformat(pack_data["created_at"]),
            updated_at=datetime.fromisoformat(pack_data["updated_at"]),
            created_by=pack_data.get("created_by", "system"),
            download_count=pack_data.get("download_count", 0),
            success_rate=pack_data.get("success_rate", 0.0),
            user_rating=pack_data.get("user_rating", 0.0)
        )
    
    async def _store_pack(self, pack: CuratedPack) -> None:
        """Store pack in database"""
        
        if not self.db_pool:
            return
        
        try:
            insert_query = """
                INSERT INTO curated_packs (
                    pack_id, pack_name, pack_type, industry_vertical,
                    complexity, version, status, pack_data, created_by
                ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9)
                ON CONFLICT (pack_id) 
                DO UPDATE SET 
                    pack_data = EXCLUDED.pack_data,
                    updated_at = NOW()
            """
            
            async with self.db_pool.acquire() as conn:
                await conn.execute(
                    insert_query,
                    pack.pack_id,
                    pack.pack_name,
                    pack.pack_type.value,
                    pack.industry_vertical.value,
                    pack.complexity.value,
                    pack.version,
                    pack.status.value,
                    json.dumps(pack.to_dict()),
                    pack.created_by
                )
        except Exception:
            # Log error but don't fail
            pass
    
    async def _record_pack_installation(
        self,
        installation_id: str,
        pack_id: str,
        tenant_id: int,
        installed_by: str,
        installed_workflows: List[Dict[str, Any]],
        applied_policies: List[Dict[str, Any]]
    ) -> None:
        """Record pack installation in database"""
        
        if not self.db_pool:
            return
        
        try:
            insert_query = """
                INSERT INTO pack_installations (
                    installation_id, pack_id, tenant_id, installed_by,
                    installed_workflows, applied_policies, installed_at
                ) VALUES ($1, $2, $3, $4, $5, $6, $7)
            """
            
            async with self.db_pool.acquire() as conn:
                await conn.execute(
                    insert_query,
                    installation_id,
                    pack_id,
                    tenant_id,
                    installed_by,
                    json.dumps(installed_workflows),
                    json.dumps(applied_policies),
                    datetime.now(timezone.utc)
                )
        except Exception:
            pass


# Database schema for curated packs
CURATED_PACKS_SCHEMA_SQL = """
-- Curated packs
CREATE TABLE IF NOT EXISTS curated_packs (
    pack_id VARCHAR(100) PRIMARY KEY,
    pack_name VARCHAR(200) NOT NULL,
    pack_type VARCHAR(50) NOT NULL,
    industry_vertical VARCHAR(50) NOT NULL,
    complexity VARCHAR(20) NOT NULL,
    version VARCHAR(20) NOT NULL,
    status VARCHAR(20) NOT NULL DEFAULT 'draft',
    pack_data JSONB NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    created_by VARCHAR(255) NOT NULL,
    
    CONSTRAINT chk_pack_type CHECK (pack_type IN ('industry_starter', 'use_case_bundle', 'compliance_pack', 'integration_pack', 'best_practices', 'demo_showcase')),
    CONSTRAINT chk_industry CHECK (industry_vertical IN ('SaaS', 'Banking', 'Insurance', 'Healthcare', 'E-commerce', 'FinTech', 'Manufacturing', 'Universal')),
    CONSTRAINT chk_complexity CHECK (complexity IN ('beginner', 'intermediate', 'advanced', 'expert')),
    CONSTRAINT chk_status CHECK (status IN ('draft', 'review', 'published', 'deprecated', 'archived'))
);

-- Pack installations
CREATE TABLE IF NOT EXISTS pack_installations (
    installation_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    pack_id VARCHAR(100) NOT NULL REFERENCES curated_packs(pack_id),
    tenant_id INTEGER NOT NULL,
    installed_by VARCHAR(255) NOT NULL,
    installed_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    installed_workflows JSONB,
    applied_policies JSONB,
    success_rate DECIMAL(3,2) DEFAULT 0,
    
    CONSTRAINT uq_pack_tenant UNIQUE (pack_id, tenant_id)
);

-- Indexes
CREATE INDEX IF NOT EXISTS idx_packs_industry ON curated_packs (industry_vertical, status);
CREATE INDEX IF NOT EXISTS idx_packs_type ON curated_packs (pack_type, status);
CREATE INDEX IF NOT EXISTS idx_packs_complexity ON curated_packs (complexity, status);
CREATE INDEX IF NOT EXISTS idx_installations_tenant ON pack_installations (tenant_id);
CREATE INDEX IF NOT EXISTS idx_installations_pack ON pack_installations (pack_id, installed_at DESC);
"""
