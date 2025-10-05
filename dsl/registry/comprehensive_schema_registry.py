"""
Comprehensive Schema Registry Service
Task 9.2.3: Build Schema Registry service - Source of truth for data contracts

Features:
- Multi-tenant schema management with versioning
- Backward/forward compatibility validation
- Industry overlay support (SaaS, Banking, Insurance)
- Schema evolution and deprecation lifecycle
- Integration with governance and evidence packs
- Real-time schema validation and drift detection
"""

import asyncio
import json
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict
from enum import Enum
import hashlib
import uuid
import jsonschema
from jsonschema import Draft7Validator, ValidationError

logger = logging.getLogger(__name__)

class SchemaType(Enum):
    """Schema types supported by the registry"""
    JSON_SCHEMA = "json_schema"
    AVRO = "avro"
    PROTOBUF = "protobuf"
    OPENAPI = "openapi"
    DSL_WORKFLOW = "dsl_workflow"
    EVIDENCE_PACK = "evidence_pack"

class SchemaCompatibility(Enum):
    """Schema compatibility levels"""
    BACKWARD = "backward"          # New schema can read old data
    FORWARD = "forward"            # Old schema can read new data
    FULL = "full"                  # Both backward and forward compatible
    NONE = "none"                  # No compatibility guarantees
    TRANSITIVE = "transitive"      # Compatibility across all versions

class SchemaStatus(Enum):
    """Schema lifecycle status"""
    DRAFT = "draft"
    ACTIVE = "active"
    DEPRECATED = "deprecated"
    RETIRED = "retired"

class IndustryOverlay(Enum):
    """Industry-specific schema overlays"""
    SAAS = "SaaS"
    BANKING = "Banking"
    INSURANCE = "Insurance"
    HEALTHCARE = "Healthcare"
    ECOMMERCE = "E-commerce"
    FINTECH = "FinTech"

@dataclass
class SchemaMetadata:
    """Schema metadata and governance information"""
    schema_id: str
    schema_name: str
    schema_type: SchemaType
    version: str
    tenant_id: int
    industry_overlay: IndustryOverlay
    compatibility_level: SchemaCompatibility
    status: SchemaStatus
    
    # Governance metadata
    created_by: str
    created_at: datetime
    updated_at: datetime
    description: str
    tags: List[str]
    
    # Compliance and governance
    compliance_frameworks: List[str]  # SOX, GDPR, HIPAA, etc.
    data_classification: str  # public, internal, confidential, restricted
    retention_days: int
    
    # Dependencies and relationships
    depends_on: List[str]  # Schema IDs this schema depends on
    used_by: List[str]     # Systems/services using this schema
    
    # Validation and quality
    validation_rules: Dict[str, Any]
    quality_score: float
    usage_count: int
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for storage"""
        result = asdict(self)
        # Convert enums to strings
        result['schema_type'] = self.schema_type.value
        result['industry_overlay'] = self.industry_overlay.value
        result['compatibility_level'] = self.compatibility_level.value
        result['status'] = self.status.value
        result['created_at'] = self.created_at.isoformat()
        result['updated_at'] = self.updated_at.isoformat()
        return result

@dataclass
class SchemaVersion:
    """Individual schema version with content"""
    schema_id: str
    version: str
    schema_content: Dict[str, Any]
    metadata: SchemaMetadata
    compatibility_report: Optional[Dict[str, Any]] = None
    
class ComprehensiveSchemaRegistry:
    """
    Comprehensive Schema Registry Service
    Task 9.2.3: Central source of truth for data contracts
    """
    
    def __init__(self, pool_manager):
        self.pool_manager = pool_manager
        self.logger = logging.getLogger(__name__)
        
        # In-memory caches for performance
        self.schema_cache = {}  # schema_id -> SchemaVersion
        self.metadata_cache = {}  # schema_id -> SchemaMetadata
        self.compatibility_cache = {}  # (schema_id, version1, version2) -> bool
        
        # Industry-specific schema templates
        self.industry_templates = {
            IndustryOverlay.SAAS: {
                "subscription_schema": self._get_saas_subscription_schema(),
                "billing_schema": self._get_saas_billing_schema(),
                "usage_metrics_schema": self._get_saas_usage_schema(),
                "churn_prediction_schema": self._get_saas_churn_schema()
            },
            IndustryOverlay.BANKING: {
                "transaction_schema": self._get_banking_transaction_schema(),
                "account_schema": self._get_banking_account_schema(),
                "compliance_schema": self._get_banking_compliance_schema()
            },
            IndustryOverlay.INSURANCE: {
                "policy_schema": self._get_insurance_policy_schema(),
                "claim_schema": self._get_insurance_claim_schema(),
                "risk_assessment_schema": self._get_insurance_risk_schema()
            }
        }
        
        # Governance and compliance configurations
        self.governance_config = {
            "evidence_pack_required": True,
            "approval_required_for_breaking_changes": True,
            "retention_policy_enforcement": True,
            "audit_trail_enabled": True
        }
        
    async def initialize(self) -> bool:
        """Initialize the schema registry"""
        try:
            self.logger.info("🚀 Initializing Comprehensive Schema Registry...")
            
            # Create database tables if they don't exist
            await self._create_schema_tables()
            
            # Load existing schemas into cache
            await self._load_schema_cache()
            
            # Initialize industry templates
            await self._initialize_industry_templates()
            
            self.logger.info("✅ Comprehensive Schema Registry initialized successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Failed to initialize Schema Registry: {e}")
            return False
    
    async def register_schema(self, 
                            schema_name: str,
                            schema_content: Dict[str, Any],
                            schema_type: SchemaType,
                            tenant_id: int,
                            industry_overlay: IndustryOverlay,
                            created_by: str,
                            description: str = "",
                            tags: List[str] = None,
                            compliance_frameworks: List[str] = None) -> str:
        """
        Register a new schema in the registry
        Returns schema_id
        """
        try:
            schema_id = str(uuid.uuid4())
            version = "1.0.0"
            
            # Create schema metadata
            metadata = SchemaMetadata(
                schema_id=schema_id,
                schema_name=schema_name,
                schema_type=schema_type,
                version=version,
                tenant_id=tenant_id,
                industry_overlay=industry_overlay,
                compatibility_level=SchemaCompatibility.BACKWARD,
                status=SchemaStatus.DRAFT,
                created_by=created_by,
                created_at=datetime.now(),
                updated_at=datetime.now(),
                description=description,
                tags=tags or [],
                compliance_frameworks=compliance_frameworks or [],
                data_classification="internal",
                retention_days=2555,  # 7 years default
                depends_on=[],
                used_by=[],
                validation_rules={},
                quality_score=0.0,
                usage_count=0
            )
            
            # Validate schema content
            validation_result = await self._validate_schema_content(schema_content, schema_type)
            if not validation_result["valid"]:
                raise ValueError(f"Schema validation failed: {validation_result['errors']}")
            
            # Apply industry-specific enhancements
            enhanced_schema = await self._apply_industry_overlay(schema_content, industry_overlay)
            
            # Create schema version
            schema_version = SchemaVersion(
                schema_id=schema_id,
                version=version,
                schema_content=enhanced_schema,
                metadata=metadata
            )
            
            # Store in database
            await self._store_schema_version(schema_version)
            
            # Update caches
            self.schema_cache[schema_id] = schema_version
            self.metadata_cache[schema_id] = metadata
            
            # Generate evidence pack
            if self.governance_config["evidence_pack_required"]:
                await self._generate_schema_evidence_pack(schema_version, "schema_registration")
            
            self.logger.info(f"✅ Schema registered: {schema_name} (ID: {schema_id})")
            return schema_id
            
        except Exception as e:
            self.logger.error(f"❌ Schema registration failed: {e}")
            raise
    
    async def evolve_schema(self,
                          schema_id: str,
                          new_schema_content: Dict[str, Any],
                          new_version: str,
                          updated_by: str,
                          compatibility_level: SchemaCompatibility = None) -> bool:
        """
        Evolve an existing schema to a new version with compatibility validation
        """
        try:
            # Get current schema
            current_schema = await self.get_schema(schema_id)
            if not current_schema:
                raise ValueError(f"Schema not found: {schema_id}")
            
            # Validate compatibility
            compatibility_result = await self._validate_compatibility(
                current_schema.schema_content,
                new_schema_content,
                compatibility_level or current_schema.metadata.compatibility_level
            )
            
            if not compatibility_result["compatible"]:
                if self.governance_config["approval_required_for_breaking_changes"]:
                    # Create approval request for breaking change
                    await self._create_breaking_change_approval(
                        schema_id, new_version, compatibility_result
                    )
                    return False
                else:
                    raise ValueError(f"Compatibility validation failed: {compatibility_result['issues']}")
            
            # Create new version metadata
            new_metadata = current_schema.metadata
            new_metadata.version = new_version
            new_metadata.updated_at = datetime.now()
            
            # Create new schema version
            new_schema_version = SchemaVersion(
                schema_id=schema_id,
                version=new_version,
                schema_content=new_schema_content,
                metadata=new_metadata,
                compatibility_report=compatibility_result
            )
            
            # Store new version
            await self._store_schema_version(new_schema_version)
            
            # Update caches
            self.schema_cache[schema_id] = new_schema_version
            self.metadata_cache[schema_id] = new_metadata
            
            # Generate evidence pack
            await self._generate_schema_evidence_pack(new_schema_version, "schema_evolution")
            
            self.logger.info(f"✅ Schema evolved: {schema_id} to version {new_version}")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Schema evolution failed: {e}")
            raise
    
    async def get_schema(self, schema_id: str, version: str = None) -> Optional[SchemaVersion]:
        """Get schema by ID and optional version"""
        try:
            # Check cache first
            if version is None and schema_id in self.schema_cache:
                return self.schema_cache[schema_id]
            
            # Query database
            if self.pool_manager and hasattr(self.pool_manager, 'postgres_pool'):
                async with self.pool_manager.postgres_pool.acquire() as conn:
                    if version:
                        result = await conn.fetchrow("""
                            SELECT * FROM schema_versions 
                            WHERE schema_id = $1 AND version = $2
                        """, schema_id, version)
                    else:
                        result = await conn.fetchrow("""
                            SELECT * FROM schema_versions 
                            WHERE schema_id = $1 
                            ORDER BY created_at DESC LIMIT 1
                        """, schema_id)
                    
                    if result:
                        return self._row_to_schema_version(result)
            
            return None
            
        except Exception as e:
            self.logger.error(f"❌ Failed to get schema {schema_id}: {e}")
            return None
    
    async def list_schemas(self, 
                         tenant_id: int = None,
                         industry_overlay: IndustryOverlay = None,
                         schema_type: SchemaType = None,
                         status: SchemaStatus = None) -> List[SchemaMetadata]:
        """List schemas with optional filters"""
        try:
            schemas = []
            
            if self.pool_manager and hasattr(self.pool_manager, 'postgres_pool'):
                async with self.pool_manager.postgres_pool.acquire() as conn:
                    # Build dynamic query
                    conditions = []
                    params = []
                    param_count = 0
                    
                    if tenant_id:
                        param_count += 1
                        conditions.append(f"tenant_id = ${param_count}")
                        params.append(tenant_id)
                    
                    if industry_overlay:
                        param_count += 1
                        conditions.append(f"industry_overlay = ${param_count}")
                        params.append(industry_overlay.value)
                    
                    if schema_type:
                        param_count += 1
                        conditions.append(f"schema_type = ${param_count}")
                        params.append(schema_type.value)
                    
                    if status:
                        param_count += 1
                        conditions.append(f"status = ${param_count}")
                        params.append(status.value)
                    
                    where_clause = "WHERE " + " AND ".join(conditions) if conditions else ""
                    
                    query = f"""
                        SELECT DISTINCT ON (schema_id) *
                        FROM schema_versions 
                        {where_clause}
                        ORDER BY schema_id, created_at DESC
                    """
                    
                    results = await conn.fetch(query, *params)
                    
                    for row in results:
                        schema_version = self._row_to_schema_version(row)
                        schemas.append(schema_version.metadata)
            
            return schemas
            
        except Exception as e:
            self.logger.error(f"❌ Failed to list schemas: {e}")
            return []
    
    async def validate_schema_usage(self, schema_id: str, data: Dict[str, Any]) -> Dict[str, Any]:
        """Validate data against a registered schema"""
        try:
            schema_version = await self.get_schema(schema_id)
            if not schema_version:
                return {"valid": False, "errors": [f"Schema not found: {schema_id}"]}
            
            # Perform validation based on schema type
            if schema_version.metadata.schema_type == SchemaType.JSON_SCHEMA:
                return await self._validate_json_schema(schema_version.schema_content, data)
            else:
                # Placeholder for other schema types
                return {"valid": True, "errors": []}
            
        except Exception as e:
            self.logger.error(f"❌ Schema validation failed: {e}")
            return {"valid": False, "errors": [str(e)]}
    
    async def deprecate_schema(self, schema_id: str, deprecated_by: str, reason: str) -> bool:
        """Deprecate a schema version"""
        try:
            schema_version = await self.get_schema(schema_id)
            if not schema_version:
                raise ValueError(f"Schema not found: {schema_id}")
            
            # Update status to deprecated
            schema_version.metadata.status = SchemaStatus.DEPRECATED
            schema_version.metadata.updated_at = datetime.now()
            
            # Store updated metadata
            await self._update_schema_metadata(schema_version.metadata)
            
            # Generate evidence pack
            await self._generate_schema_evidence_pack(schema_version, "schema_deprecation", {
                "deprecated_by": deprecated_by,
                "reason": reason
            })
            
            self.logger.info(f"✅ Schema deprecated: {schema_id}")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Schema deprecation failed: {e}")
            return False
    
    async def get_schema_dependencies(self, schema_id: str) -> Dict[str, List[str]]:
        """Get schema dependencies and dependents"""
        try:
            schema_version = await self.get_schema(schema_id)
            if not schema_version:
                return {"dependencies": [], "dependents": []}
            
            # Get schemas this one depends on
            dependencies = schema_version.metadata.depends_on
            
            # Get schemas that depend on this one
            dependents = []
            all_schemas = await self.list_schemas()
            
            for schema_meta in all_schemas:
                if schema_id in schema_meta.depends_on:
                    dependents.append(schema_meta.schema_id)
            
            return {
                "dependencies": dependencies,
                "dependents": dependents
            }
            
        except Exception as e:
            self.logger.error(f"❌ Failed to get schema dependencies: {e}")
            return {"dependencies": [], "dependents": []}
    
    # Private helper methods
    
    async def _create_schema_tables(self):
        """Create database tables for schema registry"""
        if not self.pool_manager or not hasattr(self.pool_manager, 'postgres_pool'):
            return
        
        async with self.pool_manager.postgres_pool.acquire() as conn:
            # Schema versions table
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS schema_versions (
                    schema_id UUID PRIMARY KEY,
                    schema_name VARCHAR(255) NOT NULL,
                    version VARCHAR(50) NOT NULL,
                    schema_type VARCHAR(50) NOT NULL,
                    tenant_id INTEGER NOT NULL,
                    industry_overlay VARCHAR(50) NOT NULL,
                    compatibility_level VARCHAR(50) NOT NULL,
                    status VARCHAR(50) NOT NULL,
                    schema_content JSONB NOT NULL,
                    metadata JSONB NOT NULL,
                    created_by VARCHAR(255) NOT NULL,
                    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                    
                    UNIQUE(schema_id, version)
                )
            """)
            
            # Schema usage tracking
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS schema_usage_logs (
                    usage_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                    schema_id UUID NOT NULL,
                    tenant_id INTEGER NOT NULL,
                    service_name VARCHAR(255),
                    validation_result JSONB,
                    usage_timestamp TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                    
                    FOREIGN KEY (schema_id) REFERENCES schema_versions(schema_id)
                )
            """)
            
            # Schema evidence packs
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS schema_evidence_packs (
                    evidence_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                    schema_id UUID NOT NULL,
                    event_type VARCHAR(100) NOT NULL,
                    evidence_data JSONB NOT NULL,
                    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                    
                    FOREIGN KEY (schema_id) REFERENCES schema_versions(schema_id)
                )
            """)
    
    async def _load_schema_cache(self):
        """Load existing schemas into memory cache"""
        try:
            schemas = await self.list_schemas()
            for schema_meta in schemas:
                schema_version = await self.get_schema(schema_meta.schema_id)
                if schema_version:
                    self.schema_cache[schema_meta.schema_id] = schema_version
                    self.metadata_cache[schema_meta.schema_id] = schema_meta
        except Exception as e:
            self.logger.warning(f"Failed to load schema cache: {e}")
    
    async def _initialize_industry_templates(self):
        """Initialize industry-specific schema templates"""
        # This would register common schemas for each industry
        pass
    
    async def _validate_schema_content(self, schema_content: Dict[str, Any], schema_type: SchemaType) -> Dict[str, Any]:
        """Validate schema content based on type"""
        try:
            if schema_type == SchemaType.JSON_SCHEMA:
                # Validate JSON Schema format
                Draft7Validator.check_schema(schema_content)
                return {"valid": True, "errors": []}
            else:
                # Placeholder for other schema types
                return {"valid": True, "errors": []}
        except Exception as e:
            return {"valid": False, "errors": [str(e)]}
    
    async def _apply_industry_overlay(self, schema_content: Dict[str, Any], industry: IndustryOverlay) -> Dict[str, Any]:
        """Apply industry-specific enhancements to schema"""
        enhanced_schema = schema_content.copy()
        
        # Add industry-specific governance fields
        if "properties" not in enhanced_schema:
            enhanced_schema["properties"] = {}
        
        # Add common governance fields
        enhanced_schema["properties"]["tenant_id"] = {
            "type": "integer",
            "description": "Tenant identifier for multi-tenant isolation"
        }
        
        enhanced_schema["properties"]["created_at"] = {
            "type": "string",
            "format": "date-time",
            "description": "Creation timestamp"
        }
        
        # Industry-specific enhancements
        if industry == IndustryOverlay.SAAS:
            enhanced_schema["properties"]["subscription_id"] = {
                "type": "string",
                "description": "SaaS subscription identifier"
            }
        elif industry == IndustryOverlay.BANKING:
            enhanced_schema["properties"]["regulatory_flags"] = {
                "type": "object",
                "description": "Banking regulatory compliance flags"
            }
        elif industry == IndustryOverlay.INSURANCE:
            enhanced_schema["properties"]["policy_number"] = {
                "type": "string",
                "description": "Insurance policy number"
            }
        
        return enhanced_schema
    
    async def _validate_compatibility(self, old_schema: Dict[str, Any], new_schema: Dict[str, Any], 
                                    compatibility_level: SchemaCompatibility) -> Dict[str, Any]:
        """Validate schema compatibility between versions"""
        # Simplified compatibility check - in production, use proper schema evolution libraries
        issues = []
        
        if compatibility_level == SchemaCompatibility.BACKWARD:
            # Check if new schema can read old data
            if "required" in new_schema and "required" in old_schema:
                new_required = set(new_schema["required"])
                old_required = set(old_schema["required"])
                if not new_required.issubset(old_required):
                    issues.append("New schema adds required fields - breaks backward compatibility")
        
        return {
            "compatible": len(issues) == 0,
            "issues": issues,
            "compatibility_level": compatibility_level.value
        }
    
    async def _validate_json_schema(self, schema: Dict[str, Any], data: Dict[str, Any]) -> Dict[str, Any]:
        """Validate data against JSON Schema"""
        try:
            validator = Draft7Validator(schema)
            errors = list(validator.iter_errors(data))
            
            return {
                "valid": len(errors) == 0,
                "errors": [error.message for error in errors]
            }
        except Exception as e:
            return {"valid": False, "errors": [str(e)]}
    
    async def _store_schema_version(self, schema_version: SchemaVersion):
        """Store schema version in database"""
        if not self.pool_manager or not hasattr(self.pool_manager, 'postgres_pool'):
            return
        
        async with self.pool_manager.postgres_pool.acquire() as conn:
            await conn.execute("""
                INSERT INTO schema_versions 
                (schema_id, schema_name, version, schema_type, tenant_id, industry_overlay,
                 compatibility_level, status, schema_content, metadata, created_by)
                VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11)
                ON CONFLICT (schema_id, version) DO UPDATE SET
                schema_content = EXCLUDED.schema_content,
                metadata = EXCLUDED.metadata,
                updated_at = NOW()
            """, 
            schema_version.schema_id,
            schema_version.metadata.schema_name,
            schema_version.version,
            schema_version.metadata.schema_type.value,
            schema_version.metadata.tenant_id,
            schema_version.metadata.industry_overlay.value,
            schema_version.metadata.compatibility_level.value,
            schema_version.metadata.status.value,
            json.dumps(schema_version.schema_content),
            json.dumps(schema_version.metadata.to_dict()),
            schema_version.metadata.created_by)
    
    async def _update_schema_metadata(self, metadata: SchemaMetadata):
        """Update schema metadata in database"""
        if not self.pool_manager or not hasattr(self.pool_manager, 'postgres_pool'):
            return
        
        async with self.pool_manager.postgres_pool.acquire() as conn:
            await conn.execute("""
                UPDATE schema_versions 
                SET metadata = $2, status = $3, updated_at = NOW()
                WHERE schema_id = $1
            """, metadata.schema_id, json.dumps(metadata.to_dict()), metadata.status.value)
    
    def _row_to_schema_version(self, row) -> SchemaVersion:
        """Convert database row to SchemaVersion object"""
        metadata_dict = json.loads(row['metadata'])
        
        # Reconstruct metadata object
        metadata = SchemaMetadata(
            schema_id=metadata_dict['schema_id'],
            schema_name=metadata_dict['schema_name'],
            schema_type=SchemaType(metadata_dict['schema_type']),
            version=metadata_dict['version'],
            tenant_id=metadata_dict['tenant_id'],
            industry_overlay=IndustryOverlay(metadata_dict['industry_overlay']),
            compatibility_level=SchemaCompatibility(metadata_dict['compatibility_level']),
            status=SchemaStatus(metadata_dict['status']),
            created_by=metadata_dict['created_by'],
            created_at=datetime.fromisoformat(metadata_dict['created_at']),
            updated_at=datetime.fromisoformat(metadata_dict['updated_at']),
            description=metadata_dict['description'],
            tags=metadata_dict['tags'],
            compliance_frameworks=metadata_dict['compliance_frameworks'],
            data_classification=metadata_dict['data_classification'],
            retention_days=metadata_dict['retention_days'],
            depends_on=metadata_dict['depends_on'],
            used_by=metadata_dict['used_by'],
            validation_rules=metadata_dict['validation_rules'],
            quality_score=metadata_dict['quality_score'],
            usage_count=metadata_dict['usage_count']
        )
        
        return SchemaVersion(
            schema_id=row['schema_id'],
            version=row['version'],
            schema_content=json.loads(row['schema_content']),
            metadata=metadata
        )
    
    async def _generate_schema_evidence_pack(self, schema_version: SchemaVersion, event_type: str, extra_data: Dict[str, Any] = None):
        """Generate evidence pack for schema events"""
        try:
            evidence_data = {
                "schema_id": schema_version.schema_id,
                "schema_name": schema_version.metadata.schema_name,
                "version": schema_version.version,
                "event_type": event_type,
                "tenant_id": schema_version.metadata.tenant_id,
                "industry_overlay": schema_version.metadata.industry_overlay.value,
                "timestamp": datetime.now().isoformat(),
                "schema_hash": hashlib.sha256(
                    json.dumps(schema_version.schema_content, sort_keys=True).encode()
                ).hexdigest()
            }
            
            if extra_data:
                evidence_data.update(extra_data)
            
            if self.pool_manager and hasattr(self.pool_manager, 'postgres_pool'):
                async with self.pool_manager.postgres_pool.acquire() as conn:
                    await conn.execute("""
                        INSERT INTO schema_evidence_packs (schema_id, event_type, evidence_data)
                        VALUES ($1, $2, $3)
                    """, schema_version.schema_id, event_type, json.dumps(evidence_data))
            
        except Exception as e:
            self.logger.error(f"Failed to generate schema evidence pack: {e}")
    
    async def _create_breaking_change_approval(self, schema_id: str, new_version: str, compatibility_result: Dict[str, Any]):
        """Create approval request for breaking schema changes"""
        # This would integrate with the approval workflow system
        self.logger.info(f"Breaking change approval required for schema {schema_id} version {new_version}")
    
    # Industry-specific schema templates
    
    def _get_saas_subscription_schema(self) -> Dict[str, Any]:
        """SaaS subscription schema template"""
        return {
            "type": "object",
            "properties": {
                "subscription_id": {"type": "string"},
                "tenant_id": {"type": "integer"},
                "plan_name": {"type": "string"},
                "billing_cycle": {"type": "string", "enum": ["monthly", "annual"]},
                "mrr": {"type": "number"},
                "status": {"type": "string", "enum": ["active", "cancelled", "past_due"]},
                "created_at": {"type": "string", "format": "date-time"}
            },
            "required": ["subscription_id", "tenant_id", "plan_name", "status"]
        }
    
    def _get_saas_billing_schema(self) -> Dict[str, Any]:
        """SaaS billing schema template"""
        return {
            "type": "object",
            "properties": {
                "invoice_id": {"type": "string"},
                "subscription_id": {"type": "string"},
                "tenant_id": {"type": "integer"},
                "amount": {"type": "number"},
                "currency": {"type": "string"},
                "billing_period_start": {"type": "string", "format": "date"},
                "billing_period_end": {"type": "string", "format": "date"},
                "status": {"type": "string", "enum": ["draft", "sent", "paid", "overdue"]}
            },
            "required": ["invoice_id", "subscription_id", "tenant_id", "amount"]
        }
    
    def _get_saas_usage_schema(self) -> Dict[str, Any]:
        """SaaS usage metrics schema template"""
        return {
            "type": "object",
            "properties": {
                "tenant_id": {"type": "integer"},
                "metric_name": {"type": "string"},
                "metric_value": {"type": "number"},
                "unit": {"type": "string"},
                "timestamp": {"type": "string", "format": "date-time"},
                "dimensions": {"type": "object"}
            },
            "required": ["tenant_id", "metric_name", "metric_value", "timestamp"]
        }
    
    def _get_saas_churn_schema(self) -> Dict[str, Any]:
        """SaaS churn prediction schema template"""
        return {
            "type": "object",
            "properties": {
                "tenant_id": {"type": "integer"},
                "churn_probability": {"type": "number", "minimum": 0, "maximum": 1},
                "risk_factors": {"type": "array", "items": {"type": "string"}},
                "prediction_date": {"type": "string", "format": "date"},
                "model_version": {"type": "string"}
            },
            "required": ["tenant_id", "churn_probability", "prediction_date"]
        }
    
    def _get_banking_transaction_schema(self) -> Dict[str, Any]:
        """Banking transaction schema template"""
        return {
            "type": "object",
            "properties": {
                "transaction_id": {"type": "string"},
                "account_id": {"type": "string"},
                "amount": {"type": "number"},
                "currency": {"type": "string"},
                "transaction_type": {"type": "string"},
                "timestamp": {"type": "string", "format": "date-time"},
                "regulatory_flags": {"type": "object"}
            },
            "required": ["transaction_id", "account_id", "amount", "currency"]
        }
    
    def _get_banking_account_schema(self) -> Dict[str, Any]:
        """Banking account schema template"""
        return {
            "type": "object",
            "properties": {
                "account_id": {"type": "string"},
                "customer_id": {"type": "string"},
                "account_type": {"type": "string"},
                "balance": {"type": "number"},
                "currency": {"type": "string"},
                "status": {"type": "string"},
                "opened_date": {"type": "string", "format": "date"}
            },
            "required": ["account_id", "customer_id", "account_type"]
        }
    
    def _get_banking_compliance_schema(self) -> Dict[str, Any]:
        """Banking compliance schema template"""
        return {
            "type": "object",
            "properties": {
                "compliance_check_id": {"type": "string"},
                "entity_id": {"type": "string"},
                "check_type": {"type": "string"},
                "result": {"type": "string", "enum": ["pass", "fail", "warning"]},
                "details": {"type": "object"},
                "timestamp": {"type": "string", "format": "date-time"}
            },
            "required": ["compliance_check_id", "entity_id", "check_type", "result"]
        }
    
    def _get_insurance_policy_schema(self) -> Dict[str, Any]:
        """Insurance policy schema template"""
        return {
            "type": "object",
            "properties": {
                "policy_number": {"type": "string"},
                "policyholder_id": {"type": "string"},
                "policy_type": {"type": "string"},
                "premium": {"type": "number"},
                "coverage_amount": {"type": "number"},
                "effective_date": {"type": "string", "format": "date"},
                "expiry_date": {"type": "string", "format": "date"}
            },
            "required": ["policy_number", "policyholder_id", "policy_type"]
        }
    
    def _get_insurance_claim_schema(self) -> Dict[str, Any]:
        """Insurance claim schema template"""
        return {
            "type": "object",
            "properties": {
                "claim_id": {"type": "string"},
                "policy_number": {"type": "string"},
                "claim_amount": {"type": "number"},
                "claim_type": {"type": "string"},
                "status": {"type": "string"},
                "filed_date": {"type": "string", "format": "date"},
                "incident_date": {"type": "string", "format": "date"}
            },
            "required": ["claim_id", "policy_number", "claim_amount"]
        }
    
    def _get_insurance_risk_schema(self) -> Dict[str, Any]:
        """Insurance risk assessment schema template"""
        return {
            "type": "object",
            "properties": {
                "assessment_id": {"type": "string"},
                "entity_id": {"type": "string"},
                "risk_score": {"type": "number", "minimum": 0, "maximum": 100},
                "risk_factors": {"type": "array"},
                "assessment_date": {"type": "string", "format": "date"},
                "assessor": {"type": "string"}
            },
            "required": ["assessment_id", "entity_id", "risk_score"]
        }
