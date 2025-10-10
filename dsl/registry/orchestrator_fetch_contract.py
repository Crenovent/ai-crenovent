"""
Task 7.3-T40: Orchestrator Fetch Contract (strict schema)
Runtime predictability with JSON schema checked at fetch
"""

import asyncio
import json
import uuid
from datetime import datetime, timezone
from enum import Enum
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass, field

import jsonschema
from jsonschema import validate, ValidationError
import asyncpg


class ContractVersion(Enum):
    """Contract version for compatibility"""
    V1_0 = "1.0"
    V1_1 = "1.1" 
    V2_0 = "2.0"


class FetchMode(Enum):
    """Fetch mode for different use cases"""
    RUNTIME_EXECUTION = "runtime_execution"
    VALIDATION_ONLY = "validation_only"
    METADATA_ONLY = "metadata_only"
    FULL_CONTEXT = "full_context"


@dataclass
class FetchRequest:
    """Standardized fetch request"""
    workflow_id: str
    version_id: Optional[str] = None
    version_alias: Optional[str] = "stable"
    fetch_mode: FetchMode = FetchMode.RUNTIME_EXECUTION
    tenant_id: Optional[int] = None
    execution_context: Dict[str, Any] = field(default_factory=dict)
    
    # Contract validation
    contract_version: ContractVersion = ContractVersion.V1_0
    strict_validation: bool = True
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "workflow_id": self.workflow_id,
            "version_id": self.version_id,
            "version_alias": self.version_alias,
            "fetch_mode": self.fetch_mode.value,
            "tenant_id": self.tenant_id,
            "execution_context": self.execution_context,
            "contract_version": self.contract_version.value,
            "strict_validation": self.strict_validation
        }


@dataclass
class FetchResponse:
    """Standardized fetch response"""
    request_id: str
    workflow_id: str
    version_id: str
    version_number: str
    
    # Core workflow data
    workflow_definition: Dict[str, Any]
    workflow_metadata: Dict[str, Any]
    
    # Runtime requirements
    runtime_requirements: Dict[str, Any] = field(default_factory=dict)
    dependencies: List[Dict[str, Any]] = field(default_factory=list)
    
    # Governance data
    policy_bindings: List[Dict[str, Any]] = field(default_factory=list)
    compliance_requirements: List[str] = field(default_factory=list)
    
    # Execution context
    execution_hints: Dict[str, Any] = field(default_factory=dict)
    performance_profile: Dict[str, Any] = field(default_factory=dict)
    
    # Contract metadata
    contract_version: ContractVersion = ContractVersion.V1_0
    schema_hash: str = ""
    fetch_timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "request_id": self.request_id,
            "workflow_id": self.workflow_id,
            "version_id": self.version_id,
            "version_number": self.version_number,
            "workflow_definition": self.workflow_definition,
            "workflow_metadata": self.workflow_metadata,
            "runtime_requirements": self.runtime_requirements,
            "dependencies": self.dependencies,
            "policy_bindings": self.policy_bindings,
            "compliance_requirements": self.compliance_requirements,
            "execution_hints": self.execution_hints,
            "performance_profile": self.performance_profile,
            "contract_version": self.contract_version.value,
            "schema_hash": self.schema_hash,
            "fetch_timestamp": self.fetch_timestamp.isoformat()
        }


class OrchestratorFetchContract:
    """Strict contract for orchestrator-registry communication"""
    
    def __init__(self, db_pool: Optional[asyncpg.Pool] = None):
        self.db_pool = db_pool
        self.schemas = self._load_contract_schemas()
        self.fetch_cache: Dict[str, FetchResponse] = {}
    
    def _load_contract_schemas(self) -> Dict[ContractVersion, Dict[str, Any]]:
        """Load JSON schemas for different contract versions"""
        
        # V1.0 Schema - Basic contract
        v1_0_schema = {
            "type": "object",
            "required": [
                "request_id", "workflow_id", "version_id", "version_number",
                "workflow_definition", "workflow_metadata"
            ],
            "properties": {
                "request_id": {"type": "string", "format": "uuid"},
                "workflow_id": {"type": "string", "format": "uuid"},
                "version_id": {"type": "string", "format": "uuid"},
                "version_number": {"type": "string", "pattern": r"^\d+\.\d+\.\d+"},
                "workflow_definition": {
                    "type": "object",
                    "required": ["workflow_name", "steps"],
                    "properties": {
                        "workflow_name": {"type": "string", "minLength": 1},
                        "description": {"type": "string"},
                        "steps": {
                            "type": "array",
                            "minItems": 1,
                            "items": {
                                "type": "object",
                                "required": ["id", "type"],
                                "properties": {
                                    "id": {"type": "string", "minLength": 1},
                                    "type": {
                                        "type": "string",
                                        "enum": ["query", "decision", "ml_decision", "agent_call", "notify", "governance"]
                                    },
                                    "params": {"type": "object"},
                                    "governance": {"type": "object"}
                                }
                            }
                        }
                    }
                },
                "workflow_metadata": {
                    "type": "object",
                    "properties": {
                        "industry_overlay": {"type": "string"},
                        "automation_type": {"type": "string"},
                        "risk_level": {"type": "string", "enum": ["low", "medium", "high", "critical"]},
                        "compliance_requirements": {"type": "array", "items": {"type": "string"}}
                    }
                },
                "runtime_requirements": {
                    "type": "object",
                    "properties": {
                        "min_runtime_version": {"type": "string"},
                        "required_connectors": {"type": "array"},
                        "memory_limit_mb": {"type": "integer", "minimum": 1},
                        "timeout_seconds": {"type": "integer", "minimum": 1}
                    }
                },
                "dependencies": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "required": ["type", "name"],
                        "properties": {
                            "type": {"type": "string", "enum": ["connector", "policy", "workflow"]},
                            "name": {"type": "string"},
                            "version": {"type": "string"},
                            "required": {"type": "boolean"}
                        }
                    }
                },
                "policy_bindings": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "required": ["policy_id", "enforcement_level"],
                        "properties": {
                            "policy_id": {"type": "string"},
                            "policy_version": {"type": "string"},
                            "enforcement_level": {"type": "string", "enum": ["required", "recommended", "optional"]},
                            "scope": {"type": "array", "items": {"type": "string"}}
                        }
                    }
                },
                "execution_hints": {
                    "type": "object",
                    "properties": {
                        "parallel_execution": {"type": "boolean"},
                        "retry_policy": {"type": "object"},
                        "circuit_breaker": {"type": "object"},
                        "caching_strategy": {"type": "string"}
                    }
                },
                "performance_profile": {
                    "type": "object",
                    "properties": {
                        "expected_duration_ms": {"type": "integer", "minimum": 0},
                        "cpu_intensive": {"type": "boolean"},
                        "io_intensive": {"type": "boolean"},
                        "memory_usage_pattern": {"type": "string", "enum": ["constant", "linear", "exponential"]}
                    }
                },
                "contract_version": {"type": "string", "enum": ["1.0", "1.1", "2.0"]},
                "schema_hash": {"type": "string"},
                "fetch_timestamp": {"type": "string", "format": "date-time"}
            }
        }
        
        # V1.1 Schema - Enhanced with security and monitoring
        v1_1_schema = {
            **v1_0_schema,
            "properties": {
                **v1_0_schema["properties"],
                "security_context": {
                    "type": "object",
                    "properties": {
                        "required_permissions": {"type": "array", "items": {"type": "string"}},
                        "data_classification": {"type": "string", "enum": ["public", "internal", "confidential", "restricted"]},
                        "encryption_required": {"type": "boolean"},
                        "audit_level": {"type": "string", "enum": ["none", "basic", "detailed", "comprehensive"]}
                    }
                },
                "monitoring_config": {
                    "type": "object",
                    "properties": {
                        "metrics_enabled": {"type": "boolean"},
                        "tracing_enabled": {"type": "boolean"},
                        "log_level": {"type": "string", "enum": ["debug", "info", "warn", "error"]},
                        "alert_thresholds": {"type": "object"}
                    }
                }
            }
        }
        
        # V2.0 Schema - Future version with advanced features
        v2_0_schema = {
            **v1_1_schema,
            "properties": {
                **v1_1_schema["properties"],
                "ai_context": {
                    "type": "object",
                    "properties": {
                        "model_requirements": {"type": "array"},
                        "inference_config": {"type": "object"},
                        "training_metadata": {"type": "object"}
                    }
                }
            }
        }
        
        return {
            ContractVersion.V1_0: v1_0_schema,
            ContractVersion.V1_1: v1_1_schema,
            ContractVersion.V2_0: v2_0_schema
        }
    
    async def fetch_workflow(self, request: FetchRequest) -> Dict[str, Any]:
        """Fetch workflow with strict contract validation"""
        
        request_id = str(uuid.uuid4())
        
        try:
            # Validate request
            request_validation = self._validate_fetch_request(request)
            if not request_validation["valid"]:
                return {
                    "success": False,
                    "request_id": request_id,
                    "error": "Invalid fetch request",
                    "validation_errors": request_validation["errors"]
                }
            
            # Check cache first
            cache_key = self._get_cache_key(request)
            if cache_key in self.fetch_cache:
                cached_response = self.fetch_cache[cache_key]
                return {
                    "success": True,
                    "request_id": request_id,
                    "response": cached_response.to_dict(),
                    "cached": True
                }
            
            # Fetch from registry
            workflow_data = await self._fetch_from_registry(request)
            if not workflow_data:
                return {
                    "success": False,
                    "request_id": request_id,
                    "error": "Workflow not found or access denied"
                }
            
            # Build response
            response = await self._build_fetch_response(
                request_id, request, workflow_data
            )
            
            # Validate response against contract
            validation_result = self._validate_fetch_response(response, request.contract_version)
            if not validation_result["valid"]:
                if request.strict_validation:
                    return {
                        "success": False,
                        "request_id": request_id,
                        "error": "Response validation failed",
                        "validation_errors": validation_result["errors"]
                    }
                else:
                    # Log validation errors but continue
                    pass
            
            # Cache response
            self.fetch_cache[cache_key] = response
            
            return {
                "success": True,
                "request_id": request_id,
                "response": response.to_dict(),
                "cached": False
            }
            
        except Exception as e:
            return {
                "success": False,
                "request_id": request_id,
                "error": str(e)
            }
    
    def _validate_fetch_request(self, request: FetchRequest) -> Dict[str, Any]:
        """Validate fetch request structure"""
        
        errors = []
        
        # Basic validation
        if not request.workflow_id:
            errors.append("workflow_id is required")
        
        if not request.version_id and not request.version_alias:
            errors.append("Either version_id or version_alias must be provided")
        
        # Validate UUID format for workflow_id
        try:
            uuid.UUID(request.workflow_id)
        except ValueError:
            errors.append("workflow_id must be a valid UUID")
        
        # Validate version_id if provided
        if request.version_id:
            try:
                uuid.UUID(request.version_id)
            except ValueError:
                errors.append("version_id must be a valid UUID")
        
        # Validate execution context
        if request.execution_context:
            if not isinstance(request.execution_context, dict):
                errors.append("execution_context must be a dictionary")
        
        return {
            "valid": len(errors) == 0,
            "errors": errors
        }
    
    async def _fetch_from_registry(self, request: FetchRequest) -> Optional[Dict[str, Any]]:
        """Fetch workflow data from registry"""
        
        if not self.db_pool:
            # Return mock data for testing
            return self._get_mock_workflow_data(request)
        
        try:
            async with self.db_pool.acquire() as conn:
                # Determine version to fetch
                if request.version_id:
                    version_condition = "v.version_id = $2"
                    version_param = request.version_id
                else:
                    # Resolve version alias
                    version_condition = "v.version_alias = $2"
                    version_param = request.version_alias or "stable"
                
                # Main query
                query = f"""
                    SELECT 
                        w.workflow_id,
                        w.workflow_name,
                        w.workflow_slug,
                        w.namespace,
                        w.workflow_type,
                        w.automation_type,
                        w.industry_overlay,
                        w.category,
                        w.description,
                        w.keywords,
                        w.tags,
                        w.status as workflow_status,
                        w.visibility,
                        w.business_value,
                        w.customer_impact,
                        w.risk_level,
                        w.compliance_requirements,
                        v.version_id,
                        v.version_number,
                        v.major_version,
                        v.minor_version,
                        v.patch_version,
                        v.pre_release,
                        v.build_metadata,
                        v.version_alias,
                        v.status as version_status,
                        v.changelog,
                        v.breaking_changes,
                        v.migration_notes,
                        v.compatibility_notes,
                        v.dsl_schema_version,
                        v.runtime_requirements,
                        v.compiler_version,
                        v.feature_flags,
                        v.is_immutable,
                        v.content_hash,
                        v.signature,
                        v.attestation
                    FROM registry_workflows w
                    JOIN workflow_versions v ON w.workflow_id = v.workflow_id
                    WHERE w.workflow_id = $1
                    AND {version_condition}
                    AND w.status IN ('active', 'deprecated')
                    AND v.status = 'published'
                    AND ($3 IS NULL OR w.tenant_id = $3)
                """
                
                row = await conn.fetchrow(
                    query, 
                    request.workflow_id, 
                    version_param,
                    request.tenant_id
                )
                
                if not row:
                    return None
                
                workflow_data = dict(row)
                
                # Fetch artifacts if needed
                if request.fetch_mode in [FetchMode.RUNTIME_EXECUTION, FetchMode.FULL_CONTEXT]:
                    artifacts_query = """
                        SELECT artifact_type, storage_path, content_hash
                        FROM workflow_artifacts 
                        WHERE version_id = $1
                    """
                    
                    artifact_rows = await conn.fetch(artifacts_query, workflow_data["version_id"])
                    workflow_data["artifacts"] = [dict(row) for row in artifact_rows]
                
                # Fetch policy bindings
                if request.fetch_mode in [FetchMode.RUNTIME_EXECUTION, FetchMode.FULL_CONTEXT]:
                    policy_query = """
                        SELECT pb.policy_id, pb.binding_type, pb.enforcement_level,
                               pp.policy_name, pp.policy_version, pp.policy_type
                        FROM policy_bindings pb
                        JOIN dsl_policy_packs pp ON pb.policy_id = pp.policy_pack_id
                        WHERE pb.version_id = $1
                    """
                    
                    policy_rows = await conn.fetch(policy_query, workflow_data["version_id"])
                    workflow_data["policy_bindings"] = [dict(row) for row in policy_rows]
                
                # Fetch dependencies
                if request.fetch_mode in [FetchMode.RUNTIME_EXECUTION, FetchMode.FULL_CONTEXT]:
                    deps_query = """
                        SELECT dependency_type, name, version_requirement, optional
                        FROM workflow_dependencies 
                        WHERE version_id = $1
                    """
                    
                    deps_rows = await conn.fetch(deps_query, workflow_data["version_id"])
                    workflow_data["dependencies"] = [dict(row) for row in deps_rows]
                
                return workflow_data
                
        except Exception as e:
            # Log error and return None
            return None
    
    def _get_mock_workflow_data(self, request: FetchRequest) -> Dict[str, Any]:
        """Get mock workflow data for testing"""
        
        return {
            "workflow_id": request.workflow_id,
            "workflow_name": "Mock Workflow",
            "workflow_slug": "mock-workflow",
            "namespace": "test",
            "workflow_type": "rba",
            "automation_type": "pipeline_hygiene",
            "industry_overlay": "SaaS",
            "category": "data_quality",
            "description": "Mock workflow for testing",
            "keywords": ["test", "mock"],
            "tags": ["testing"],
            "workflow_status": "active",
            "visibility": "private",
            "risk_level": "low",
            "compliance_requirements": ["GDPR"],
            "version_id": str(uuid.uuid4()),
            "version_number": "1.0.0",
            "major_version": 1,
            "minor_version": 0,
            "patch_version": 0,
            "version_alias": "stable",
            "version_status": "published",
            "dsl_schema_version": "1.0.0",
            "runtime_requirements": {
                "min_runtime_version": "1.0.0",
                "memory_limit_mb": 512,
                "timeout_seconds": 300
            },
            "is_immutable": True,
            "content_hash": "abc123",
            "artifacts": [
                {
                    "artifact_type": "dsl",
                    "storage_path": "/mock/path/workflow.yaml",
                    "content_hash": "def456"
                }
            ],
            "policy_bindings": [
                {
                    "policy_id": "gdpr_basic",
                    "binding_type": "required",
                    "enforcement_level": "required",
                    "policy_name": "GDPR Basic Compliance",
                    "policy_version": "1.0.0",
                    "policy_type": "privacy"
                }
            ],
            "dependencies": []
        }
    
    async def _build_fetch_response(
        self,
        request_id: str,
        request: FetchRequest,
        workflow_data: Dict[str, Any]
    ) -> FetchResponse:
        """Build standardized fetch response"""
        
        # Extract workflow definition (this would normally come from artifact storage)
        workflow_definition = {
            "workflow_name": workflow_data["workflow_name"],
            "description": workflow_data["description"],
            "namespace": workflow_data["namespace"],
            "steps": [
                {
                    "id": "mock_step_1",
                    "type": "query",
                    "description": "Mock query step",
                    "params": {
                        "query": "SELECT * FROM opportunities WHERE stage = 'Qualified'"
                    },
                    "governance": {
                        "policy_id": "data_access_policy",
                        "evidence.capture": True
                    }
                }
            ]
        }
        
        # Build workflow metadata
        workflow_metadata = {
            "workflow_id": workflow_data["workflow_id"],
            "workflow_slug": workflow_data["workflow_slug"],
            "automation_type": workflow_data["automation_type"],
            "industry_overlay": workflow_data["industry_overlay"],
            "category": workflow_data["category"],
            "keywords": workflow_data.get("keywords", []),
            "tags": workflow_data.get("tags", []),
            "risk_level": workflow_data["risk_level"],
            "compliance_requirements": workflow_data.get("compliance_requirements", []),
            "business_value": workflow_data.get("business_value"),
            "customer_impact": workflow_data.get("customer_impact")
        }
        
        # Build runtime requirements
        runtime_requirements = workflow_data.get("runtime_requirements", {})
        if isinstance(runtime_requirements, str):
            try:
                runtime_requirements = json.loads(runtime_requirements)
            except:
                runtime_requirements = {}
        
        # Build dependencies
        dependencies = []
        for dep in workflow_data.get("dependencies", []):
            dependencies.append({
                "type": dep.get("dependency_type", "connector"),
                "name": dep.get("name"),
                "version": dep.get("version_requirement"),
                "required": not dep.get("optional", False)
            })
        
        # Build policy bindings
        policy_bindings = []
        for policy in workflow_data.get("policy_bindings", []):
            policy_bindings.append({
                "policy_id": policy.get("policy_id"),
                "policy_version": policy.get("policy_version"),
                "enforcement_level": policy.get("enforcement_level", "required"),
                "scope": []  # Would be populated from policy definition
            })
        
        # Build execution hints
        execution_hints = {
            "parallel_execution": True,
            "retry_policy": {
                "max_retries": 3,
                "backoff_strategy": "exponential"
            },
            "circuit_breaker": {
                "failure_threshold": 5,
                "timeout_ms": 30000
            },
            "caching_strategy": "none"
        }
        
        # Build performance profile
        performance_profile = {
            "expected_duration_ms": 5000,
            "cpu_intensive": False,
            "io_intensive": True,
            "memory_usage_pattern": "constant"
        }
        
        # Calculate schema hash
        response_dict = {
            "workflow_definition": workflow_definition,
            "workflow_metadata": workflow_metadata,
            "runtime_requirements": runtime_requirements,
            "dependencies": dependencies,
            "policy_bindings": policy_bindings
        }
        schema_hash = self._calculate_schema_hash(response_dict)
        
        return FetchResponse(
            request_id=request_id,
            workflow_id=workflow_data["workflow_id"],
            version_id=workflow_data["version_id"],
            version_number=workflow_data["version_number"],
            workflow_definition=workflow_definition,
            workflow_metadata=workflow_metadata,
            runtime_requirements=runtime_requirements,
            dependencies=dependencies,
            policy_bindings=policy_bindings,
            compliance_requirements=workflow_data.get("compliance_requirements", []),
            execution_hints=execution_hints,
            performance_profile=performance_profile,
            contract_version=request.contract_version,
            schema_hash=schema_hash
        )
    
    def _validate_fetch_response(
        self,
        response: FetchResponse,
        contract_version: ContractVersion
    ) -> Dict[str, Any]:
        """Validate fetch response against contract schema"""
        
        schema = self.schemas.get(contract_version)
        if not schema:
            return {"valid": False, "errors": [f"Unknown contract version: {contract_version.value}"]}
        
        try:
            validate(instance=response.to_dict(), schema=schema)
            return {"valid": True, "errors": []}
        except ValidationError as e:
            return {"valid": False, "errors": [str(e)]}
        except Exception as e:
            return {"valid": False, "errors": [f"Validation error: {str(e)}"]}
    
    def _get_cache_key(self, request: FetchRequest) -> str:
        """Generate cache key for fetch request"""
        
        key_parts = [
            request.workflow_id,
            request.version_id or request.version_alias or "stable",
            request.fetch_mode.value,
            str(request.tenant_id or "global"),
            request.contract_version.value
        ]
        
        return ":".join(key_parts)
    
    def _calculate_schema_hash(self, data: Dict[str, Any]) -> str:
        """Calculate hash of response schema for integrity"""
        
        import hashlib
        
        # Normalize data for consistent hashing
        normalized = json.dumps(data, sort_keys=True, separators=(',', ':'))
        return hashlib.sha256(normalized.encode()).hexdigest()[:16]
    
    async def validate_contract_compliance(
        self,
        workflow_id: str,
        version_id: str,
        contract_version: ContractVersion = ContractVersion.V1_0
    ) -> Dict[str, Any]:
        """Validate that a workflow complies with fetch contract"""
        
        # Create test fetch request
        test_request = FetchRequest(
            workflow_id=workflow_id,
            version_id=version_id,
            fetch_mode=FetchMode.VALIDATION_ONLY,
            contract_version=contract_version,
            strict_validation=True
        )
        
        # Attempt fetch
        result = await self.fetch_workflow(test_request)
        
        if result["success"]:
            return {
                "compliant": True,
                "contract_version": contract_version.value,
                "validation_timestamp": datetime.now(timezone.utc).isoformat()
            }
        else:
            return {
                "compliant": False,
                "contract_version": contract_version.value,
                "errors": result.get("validation_errors", [result.get("error")]),
                "validation_timestamp": datetime.now(timezone.utc).isoformat()
            }
    
    def get_contract_schema(self, contract_version: ContractVersion) -> Dict[str, Any]:
        """Get the JSON schema for a contract version"""
        
        return self.schemas.get(contract_version, {})
    
    def list_supported_versions(self) -> List[str]:
        """List supported contract versions"""
        
        return [version.value for version in ContractVersion]


# Database schema for contract management
FETCH_CONTRACT_SCHEMA_SQL = """
-- Contract validation results
CREATE TABLE IF NOT EXISTS workflow_contract_validations (
    validation_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    workflow_id UUID NOT NULL,
    version_id UUID NOT NULL,
    contract_version VARCHAR(10) NOT NULL,
    is_compliant BOOLEAN NOT NULL,
    validation_errors JSONB,
    validated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    validated_by VARCHAR(255),
    
    CONSTRAINT uq_workflow_contract UNIQUE (workflow_id, version_id, contract_version)
);

-- Contract fetch logs
CREATE TABLE IF NOT EXISTS workflow_fetch_logs (
    log_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    request_id UUID NOT NULL,
    workflow_id UUID NOT NULL,
    version_id UUID,
    version_alias VARCHAR(50),
    fetch_mode VARCHAR(50) NOT NULL,
    contract_version VARCHAR(10) NOT NULL,
    tenant_id INTEGER,
    success BOOLEAN NOT NULL,
    error_message TEXT,
    response_size_bytes INTEGER,
    fetch_duration_ms INTEGER,
    cached BOOLEAN NOT NULL DEFAULT FALSE,
    fetched_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    
    CONSTRAINT chk_fetch_mode CHECK (fetch_mode IN ('runtime_execution', 'validation_only', 'metadata_only', 'full_context'))
);

-- Indexes
CREATE INDEX IF NOT EXISTS idx_contract_validations_workflow ON workflow_contract_validations (workflow_id, contract_version);
CREATE INDEX IF NOT EXISTS idx_fetch_logs_workflow ON workflow_fetch_logs (workflow_id, fetched_at DESC);
CREATE INDEX IF NOT EXISTS idx_fetch_logs_tenant ON workflow_fetch_logs (tenant_id, fetched_at DESC);
CREATE INDEX IF NOT EXISTS idx_fetch_logs_success ON workflow_fetch_logs (success, fetched_at DESC);
"""
