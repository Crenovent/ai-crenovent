"""
Task 7.3-T41: Support Compiled Artifact upload (plan hash)
Determinism at runtime with stored template and hash validation
"""

import asyncio
import hashlib
import json
import uuid
from datetime import datetime, timezone
from enum import Enum
from typing import Dict, List, Optional, Any, BinaryIO, Union
from dataclasses import dataclass, field
from pathlib import Path
import tempfile

import asyncpg


class CompilationStatus(Enum):
    """Compilation status"""
    PENDING = "pending"
    COMPILING = "compiling"
    COMPILED = "compiled"
    FAILED = "failed"
    INVALID = "invalid"


class ArtifactType(Enum):
    """Types of compiled artifacts"""
    EXECUTION_PLAN = "execution_plan"
    DEPENDENCY_GRAPH = "dependency_graph"
    POLICY_BUNDLE = "policy_bundle"
    VALIDATION_RULES = "validation_rules"
    RUNTIME_CONFIG = "runtime_config"


@dataclass
class CompilationRequest:
    """Request for workflow compilation"""
    workflow_id: str
    version_id: str
    compilation_target: str = "runtime"
    optimization_level: str = "standard"
    include_debug_info: bool = False
    compiler_options: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "workflow_id": self.workflow_id,
            "version_id": self.version_id,
            "compilation_target": self.compilation_target,
            "optimization_level": self.optimization_level,
            "include_debug_info": self.include_debug_info,
            "compiler_options": self.compiler_options
        }


@dataclass
class CompiledArtifact:
    """Compiled workflow artifact"""
    artifact_id: str
    workflow_id: str
    version_id: str
    artifact_type: ArtifactType
    
    # Compilation metadata
    compiler_version: str
    compilation_target: str
    optimization_level: str
    compiled_at: datetime
    
    # Content and integrity
    content_data: Union[str, bytes, Dict[str, Any]]
    content_hash: str
    content_size_bytes: int
    
    # Validation
    schema_version: str = "1.0"
    is_deterministic: bool = True
    validation_checksum: Optional[str] = None
    
    # Dependencies
    dependencies: List[Dict[str, Any]] = field(default_factory=list)
    
    # Performance metadata
    compilation_duration_ms: int = 0
    estimated_execution_time_ms: int = 0
    memory_footprint_bytes: int = 0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "artifact_id": self.artifact_id,
            "workflow_id": self.workflow_id,
            "version_id": self.version_id,
            "artifact_type": self.artifact_type.value,
            "compiler_version": self.compiler_version,
            "compilation_target": self.compilation_target,
            "optimization_level": self.optimization_level,
            "compiled_at": self.compiled_at.isoformat(),
            "content_hash": self.content_hash,
            "content_size_bytes": self.content_size_bytes,
            "schema_version": self.schema_version,
            "is_deterministic": self.is_deterministic,
            "validation_checksum": self.validation_checksum,
            "dependencies": self.dependencies,
            "compilation_duration_ms": self.compilation_duration_ms,
            "estimated_execution_time_ms": self.estimated_execution_time_ms,
            "memory_footprint_bytes": self.memory_footprint_bytes
        }


class CompiledArtifactManager:
    """Manager for compiled workflow artifacts"""
    
    def __init__(self, db_pool: Optional[asyncpg.Pool] = None, storage_path: Optional[str] = None):
        self.db_pool = db_pool
        self.storage_path = Path(storage_path) if storage_path else Path(tempfile.gettempdir()) / "compiled_artifacts"
        self.storage_path.mkdir(parents=True, exist_ok=True)
        self.compiler_version = "1.0.0"
    
    async def compile_workflow(
        self,
        request: CompilationRequest,
        source_dsl: str,
        source_metadata: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Compile workflow and generate artifacts"""
        
        compilation_id = str(uuid.uuid4())
        start_time = datetime.now(timezone.utc)
        
        try:
            # Parse source DSL
            if source_dsl.strip().startswith('{'):
                workflow_def = json.loads(source_dsl)
            else:
                import yaml
                workflow_def = yaml.safe_load(source_dsl)
            
            # Generate execution plan
            execution_plan = await self._compile_execution_plan(
                workflow_def, request, source_metadata
            )
            
            # Generate dependency graph
            dependency_graph = await self._compile_dependency_graph(
                workflow_def, request
            )
            
            # Generate policy bundle
            policy_bundle = await self._compile_policy_bundle(
                workflow_def, request, source_metadata
            )
            
            # Generate validation rules
            validation_rules = await self._compile_validation_rules(
                workflow_def, request
            )
            
            # Generate runtime config
            runtime_config = await self._compile_runtime_config(
                workflow_def, request, source_metadata
            )
            
            # Create artifacts
            artifacts = []
            
            for artifact_type, content in [
                (ArtifactType.EXECUTION_PLAN, execution_plan),
                (ArtifactType.DEPENDENCY_GRAPH, dependency_graph),
                (ArtifactType.POLICY_BUNDLE, policy_bundle),
                (ArtifactType.VALIDATION_RULES, validation_rules),
                (ArtifactType.RUNTIME_CONFIG, runtime_config)
            ]:
                artifact = await self._create_compiled_artifact(
                    request, artifact_type, content, start_time
                )
                artifacts.append(artifact)
                
                # Store artifact
                await self._store_artifact(artifact)
            
            compilation_duration = (datetime.now(timezone.utc) - start_time).total_seconds() * 1000
            
            # Record compilation
            if self.db_pool:
                await self._record_compilation(
                    compilation_id, request, artifacts, compilation_duration, "compiled"
                )
            
            return {
                "success": True,
                "compilation_id": compilation_id,
                "artifacts": [artifact.to_dict() for artifact in artifacts],
                "compilation_duration_ms": int(compilation_duration),
                "status": "compiled"
            }
            
        except Exception as e:
            compilation_duration = (datetime.now(timezone.utc) - start_time).total_seconds() * 1000
            
            if self.db_pool:
                await self._record_compilation(
                    compilation_id, request, [], compilation_duration, "failed", str(e)
                )
            
            return {
                "success": False,
                "compilation_id": compilation_id,
                "error": str(e),
                "compilation_duration_ms": int(compilation_duration),
                "status": "failed"
            }
    
    async def _compile_execution_plan(
        self,
        workflow_def: Dict[str, Any],
        request: CompilationRequest,
        metadata: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Compile workflow into execution plan"""
        
        steps = workflow_def.get("steps", [])
        
        # Build execution graph
        execution_graph = {
            "nodes": [],
            "edges": [],
            "entry_points": [],
            "exit_points": []
        }
        
        # Process steps
        for i, step in enumerate(steps):
            step_id = step.get("id", f"step_{i}")
            
            node = {
                "id": step_id,
                "type": step.get("type", "unknown"),
                "operation": self._compile_step_operation(step, request),
                "dependencies": self._extract_step_dependencies(step),
                "governance": self._compile_step_governance(step),
                "retry_policy": self._compile_retry_policy(step),
                "timeout_ms": self._extract_timeout(step),
                "parallel_safe": self._is_parallel_safe(step)
            }
            
            execution_graph["nodes"].append(node)
            
            # Add edges based on step dependencies
            if i == 0:
                execution_graph["entry_points"].append(step_id)
            
            if i == len(steps) - 1:
                execution_graph["exit_points"].append(step_id)
            
            # Connect to previous step (simplified)
            if i > 0:
                prev_step_id = steps[i-1].get("id", f"step_{i-1}")
                execution_graph["edges"].append({
                    "from": prev_step_id,
                    "to": step_id,
                    "condition": None
                })
        
        # Optimization passes
        if request.optimization_level in ["aggressive", "maximum"]:
            execution_graph = self._optimize_execution_graph(execution_graph)
        
        return {
            "workflow_id": request.workflow_id,
            "version_id": request.version_id,
            "execution_graph": execution_graph,
            "metadata": {
                "total_steps": len(steps),
                "parallel_steps": len([n for n in execution_graph["nodes"] if n["parallel_safe"]]),
                "estimated_duration_ms": sum(n.get("timeout_ms", 5000) for n in execution_graph["nodes"]),
                "optimization_level": request.optimization_level
            }
        }
    
    async def _compile_dependency_graph(
        self,
        workflow_def: Dict[str, Any],
        request: CompilationRequest
    ) -> Dict[str, Any]:
        """Compile dependency graph"""
        
        dependencies = {
            "connectors": [],
            "policies": [],
            "workflows": [],
            "external_apis": [],
            "data_sources": []
        }
        
        steps = workflow_def.get("steps", [])
        
        for step in steps:
            step_type = step.get("type")
            params = step.get("params", {})
            
            # Extract connector dependencies
            if step_type == "query":
                data_source = params.get("data_source")
                if data_source:
                    dependencies["data_sources"].append({
                        "name": data_source,
                        "type": "database",
                        "required": True
                    })
            
            # Extract API dependencies
            elif step_type == "agent_call":
                api_endpoint = params.get("endpoint")
                if api_endpoint:
                    dependencies["external_apis"].append({
                        "endpoint": api_endpoint,
                        "method": params.get("method", "POST"),
                        "required": True
                    })
            
            # Extract policy dependencies
            governance = step.get("governance", {})
            policy_id = governance.get("policy_id")
            if policy_id:
                dependencies["policies"].append({
                    "policy_id": policy_id,
                    "enforcement_level": "required"
                })
        
        return {
            "workflow_id": request.workflow_id,
            "version_id": request.version_id,
            "dependencies": dependencies,
            "dependency_count": sum(len(deps) for deps in dependencies.values())
        }
    
    async def _compile_policy_bundle(
        self,
        workflow_def: Dict[str, Any],
        request: CompilationRequest,
        metadata: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Compile policy enforcement bundle"""
        
        policies = []
        steps = workflow_def.get("steps", [])
        
        # Extract governance policies from steps
        for step in steps:
            governance = step.get("governance", {})
            if governance:
                policy = {
                    "step_id": step.get("id"),
                    "policy_id": governance.get("policy_id"),
                    "enforcement_points": [],
                    "evidence_requirements": {},
                    "override_conditions": []
                }
                
                # Evidence capture requirements
                if governance.get("evidence.capture"):
                    policy["evidence_requirements"]["capture_enabled"] = True
                    policy["evidence_requirements"]["retention_days"] = governance.get("evidence.retention_days", 90)
                
                # Override ledger requirements
                if governance.get("override_ledger_id"):
                    policy["override_conditions"].append({
                        "ledger_id": governance.get("override_ledger_id"),
                        "approval_required": True
                    })
                
                policies.append(policy)
        
        # Industry-specific policies
        industry_overlay = metadata.get("industry_overlay", "UNIVERSAL")
        if industry_overlay != "UNIVERSAL":
            policies.extend(self._get_industry_policies(industry_overlay))
        
        return {
            "workflow_id": request.workflow_id,
            "version_id": request.version_id,
            "policies": policies,
            "industry_overlay": industry_overlay,
            "compliance_level": metadata.get("risk_level", "medium")
        }
    
    async def _compile_validation_rules(
        self,
        workflow_def: Dict[str, Any],
        request: CompilationRequest
    ) -> Dict[str, Any]:
        """Compile validation rules"""
        
        rules = {
            "input_validation": [],
            "output_validation": [],
            "state_validation": [],
            "business_rules": []
        }
        
        steps = workflow_def.get("steps", [])
        
        for step in steps:
            step_type = step.get("type")
            params = step.get("params", {})
            
            # Input validation rules
            if "input_schema" in params:
                rules["input_validation"].append({
                    "step_id": step.get("id"),
                    "schema": params["input_schema"],
                    "required": True
                })
            
            # Business logic validation
            if step_type == "decision":
                condition = params.get("condition")
                if condition:
                    rules["business_rules"].append({
                        "step_id": step.get("id"),
                        "rule_type": "decision_condition",
                        "expression": condition,
                        "validation_required": True
                    })
        
        return {
            "workflow_id": request.workflow_id,
            "version_id": request.version_id,
            "validation_rules": rules,
            "rule_count": sum(len(rule_list) for rule_list in rules.values())
        }
    
    async def _compile_runtime_config(
        self,
        workflow_def: Dict[str, Any],
        request: CompilationRequest,
        metadata: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Compile runtime configuration"""
        
        config = {
            "execution": {
                "max_concurrent_steps": 5,
                "default_timeout_ms": 30000,
                "retry_policy": {
                    "max_retries": 3,
                    "backoff_strategy": "exponential",
                    "base_delay_ms": 1000
                },
                "circuit_breaker": {
                    "failure_threshold": 5,
                    "recovery_timeout_ms": 60000
                }
            },
            "resources": {
                "memory_limit_mb": 512,
                "cpu_limit_cores": 1.0,
                "disk_limit_mb": 1024,
                "network_timeout_ms": 10000
            },
            "monitoring": {
                "metrics_enabled": True,
                "tracing_enabled": True,
                "log_level": "INFO",
                "performance_tracking": True
            },
            "security": {
                "encryption_required": False,
                "audit_level": "standard",
                "access_control": "rbac"
            }
        }
        
        # Adjust based on workflow characteristics
        step_count = len(workflow_def.get("steps", []))
        if step_count > 10:
            config["execution"]["max_concurrent_steps"] = min(10, step_count // 2)
            config["resources"]["memory_limit_mb"] = 1024
        
        # Industry-specific adjustments
        industry = metadata.get("industry_overlay", "UNIVERSAL")
        if industry in ["Banking", "Insurance", "Healthcare"]:
            config["security"]["encryption_required"] = True
            config["security"]["audit_level"] = "comprehensive"
            config["monitoring"]["log_level"] = "DEBUG"
        
        return {
            "workflow_id": request.workflow_id,
            "version_id": request.version_id,
            "runtime_config": config,
            "config_hash": self._calculate_config_hash(config)
        }
    
    async def _create_compiled_artifact(
        self,
        request: CompilationRequest,
        artifact_type: ArtifactType,
        content: Dict[str, Any],
        start_time: datetime
    ) -> CompiledArtifact:
        """Create compiled artifact with metadata"""
        
        # Serialize content
        content_json = json.dumps(content, sort_keys=True, separators=(',', ':'))
        content_bytes = content_json.encode('utf-8')
        
        # Calculate hash
        content_hash = hashlib.sha256(content_bytes).hexdigest()
        
        # Calculate validation checksum (includes metadata for determinism)
        validation_data = {
            "content_hash": content_hash,
            "compiler_version": self.compiler_version,
            "compilation_target": request.compilation_target,
            "optimization_level": request.optimization_level
        }
        validation_checksum = hashlib.md5(
            json.dumps(validation_data, sort_keys=True).encode()
        ).hexdigest()
        
        return CompiledArtifact(
            artifact_id=str(uuid.uuid4()),
            workflow_id=request.workflow_id,
            version_id=request.version_id,
            artifact_type=artifact_type,
            compiler_version=self.compiler_version,
            compilation_target=request.compilation_target,
            optimization_level=request.optimization_level,
            compiled_at=datetime.now(timezone.utc),
            content_data=content,
            content_hash=content_hash,
            content_size_bytes=len(content_bytes),
            validation_checksum=validation_checksum,
            compilation_duration_ms=int((datetime.now(timezone.utc) - start_time).total_seconds() * 1000),
            estimated_execution_time_ms=content.get("metadata", {}).get("estimated_duration_ms", 5000),
            memory_footprint_bytes=len(content_bytes) * 2  # Rough estimate
        )
    
    async def _store_artifact(self, artifact: CompiledArtifact) -> None:
        """Store compiled artifact"""
        
        # Store to filesystem
        artifact_path = self.storage_path / f"{artifact.artifact_id}.json"
        
        with open(artifact_path, 'w') as f:
            json.dump(artifact.content_data, f, indent=2)
        
        # Store metadata to database
        if self.db_pool:
            try:
                insert_query = """
                    INSERT INTO compiled_artifacts (
                        artifact_id, workflow_id, version_id, artifact_type,
                        compiler_version, compilation_target, optimization_level,
                        compiled_at, content_hash, content_size_bytes,
                        schema_version, is_deterministic, validation_checksum,
                        storage_path, compilation_duration_ms,
                        estimated_execution_time_ms, memory_footprint_bytes
                    ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14, $15, $16, $17)
                """
                
                async with self.db_pool.acquire() as conn:
                    await conn.execute(
                        insert_query,
                        artifact.artifact_id,
                        artifact.workflow_id,
                        artifact.version_id,
                        artifact.artifact_type.value,
                        artifact.compiler_version,
                        artifact.compilation_target,
                        artifact.optimization_level,
                        artifact.compiled_at,
                        artifact.content_hash,
                        artifact.content_size_bytes,
                        artifact.schema_version,
                        artifact.is_deterministic,
                        artifact.validation_checksum,
                        str(artifact_path),
                        artifact.compilation_duration_ms,
                        artifact.estimated_execution_time_ms,
                        artifact.memory_footprint_bytes
                    )
            except Exception:
                # Log error but don't fail
                pass
    
    async def get_compiled_artifact(
        self,
        workflow_id: str,
        version_id: str,
        artifact_type: ArtifactType
    ) -> Optional[CompiledArtifact]:
        """Retrieve compiled artifact"""
        
        if not self.db_pool:
            return None
        
        try:
            query = """
                SELECT * FROM compiled_artifacts 
                WHERE workflow_id = $1 AND version_id = $2 AND artifact_type = $3
                ORDER BY compiled_at DESC
                LIMIT 1
            """
            
            async with self.db_pool.acquire() as conn:
                row = await conn.fetchrow(query, workflow_id, version_id, artifact_type.value)
                
                if not row:
                    return None
                
                # Load content from storage
                storage_path = Path(row["storage_path"])
                if storage_path.exists():
                    with open(storage_path, 'r') as f:
                        content_data = json.load(f)
                else:
                    return None
                
                return CompiledArtifact(
                    artifact_id=row["artifact_id"],
                    workflow_id=row["workflow_id"],
                    version_id=row["version_id"],
                    artifact_type=ArtifactType(row["artifact_type"]),
                    compiler_version=row["compiler_version"],
                    compilation_target=row["compilation_target"],
                    optimization_level=row["optimization_level"],
                    compiled_at=row["compiled_at"],
                    content_data=content_data,
                    content_hash=row["content_hash"],
                    content_size_bytes=row["content_size_bytes"],
                    schema_version=row["schema_version"],
                    is_deterministic=row["is_deterministic"],
                    validation_checksum=row["validation_checksum"],
                    compilation_duration_ms=row["compilation_duration_ms"],
                    estimated_execution_time_ms=row["estimated_execution_time_ms"],
                    memory_footprint_bytes=row["memory_footprint_bytes"]
                )
        except Exception:
            return None
    
    async def validate_artifact_integrity(
        self,
        artifact: CompiledArtifact
    ) -> Dict[str, Any]:
        """Validate compiled artifact integrity"""
        
        # Recalculate content hash
        content_json = json.dumps(artifact.content_data, sort_keys=True, separators=(',', ':'))
        calculated_hash = hashlib.sha256(content_json.encode()).hexdigest()
        
        # Validate hash match
        hash_valid = calculated_hash == artifact.content_hash
        
        # Validate checksum
        validation_data = {
            "content_hash": artifact.content_hash,
            "compiler_version": artifact.compiler_version,
            "compilation_target": artifact.compilation_target,
            "optimization_level": artifact.optimization_level
        }
        calculated_checksum = hashlib.md5(
            json.dumps(validation_data, sort_keys=True).encode()
        ).hexdigest()
        
        checksum_valid = calculated_checksum == artifact.validation_checksum
        
        return {
            "artifact_id": artifact.artifact_id,
            "integrity_valid": hash_valid and checksum_valid,
            "hash_valid": hash_valid,
            "checksum_valid": checksum_valid,
            "calculated_hash": calculated_hash,
            "stored_hash": artifact.content_hash,
            "calculated_checksum": calculated_checksum,
            "stored_checksum": artifact.validation_checksum,
            "validation_timestamp": datetime.now(timezone.utc).isoformat()
        }
    
    def _compile_step_operation(self, step: Dict[str, Any], request: CompilationRequest) -> Dict[str, Any]:
        """Compile step into operation"""
        return {
            "type": step.get("type"),
            "params": step.get("params", {}),
            "compiled": True,
            "optimization_level": request.optimization_level
        }
    
    def _extract_step_dependencies(self, step: Dict[str, Any]) -> List[str]:
        """Extract step dependencies"""
        # Simplified dependency extraction
        return []
    
    def _compile_step_governance(self, step: Dict[str, Any]) -> Dict[str, Any]:
        """Compile step governance"""
        return step.get("governance", {})
    
    def _compile_retry_policy(self, step: Dict[str, Any]) -> Dict[str, Any]:
        """Compile retry policy for step"""
        return {
            "max_retries": 3,
            "backoff_strategy": "exponential",
            "base_delay_ms": 1000
        }
    
    def _extract_timeout(self, step: Dict[str, Any]) -> int:
        """Extract timeout for step"""
        return step.get("timeout_ms", 30000)
    
    def _is_parallel_safe(self, step: Dict[str, Any]) -> bool:
        """Check if step is safe for parallel execution"""
        return step.get("type") in ["query", "notify"]
    
    def _optimize_execution_graph(self, graph: Dict[str, Any]) -> Dict[str, Any]:
        """Optimize execution graph"""
        # Simplified optimization
        return graph
    
    def _get_industry_policies(self, industry: str) -> List[Dict[str, Any]]:
        """Get industry-specific policies"""
        return []
    
    def _calculate_config_hash(self, config: Dict[str, Any]) -> str:
        """Calculate configuration hash"""
        config_json = json.dumps(config, sort_keys=True)
        return hashlib.md5(config_json.encode()).hexdigest()
    
    async def _record_compilation(
        self,
        compilation_id: str,
        request: CompilationRequest,
        artifacts: List[CompiledArtifact],
        duration_ms: float,
        status: str,
        error_message: Optional[str] = None
    ) -> None:
        """Record compilation in database"""
        
        if not self.db_pool:
            return
        
        try:
            insert_query = """
                INSERT INTO workflow_compilations (
                    compilation_id, workflow_id, version_id, compilation_target,
                    optimization_level, compiler_version, compiled_at,
                    compilation_duration_ms, status, error_message,
                    artifact_count, total_size_bytes
                ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12)
            """
            
            total_size = sum(artifact.content_size_bytes for artifact in artifacts)
            
            async with self.db_pool.acquire() as conn:
                await conn.execute(
                    insert_query,
                    compilation_id,
                    request.workflow_id,
                    request.version_id,
                    request.compilation_target,
                    request.optimization_level,
                    self.compiler_version,
                    datetime.now(timezone.utc),
                    int(duration_ms),
                    status,
                    error_message,
                    len(artifacts),
                    total_size
                )
        except Exception:
            pass


# Database schema for compiled artifacts
COMPILED_ARTIFACTS_SCHEMA_SQL = """
-- Compiled artifacts
CREATE TABLE IF NOT EXISTS compiled_artifacts (
    artifact_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    workflow_id UUID NOT NULL,
    version_id UUID NOT NULL,
    artifact_type VARCHAR(50) NOT NULL,
    
    -- Compilation metadata
    compiler_version VARCHAR(20) NOT NULL,
    compilation_target VARCHAR(50) NOT NULL,
    optimization_level VARCHAR(20) NOT NULL,
    compiled_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    
    -- Content and integrity
    content_hash VARCHAR(64) NOT NULL,
    content_size_bytes INTEGER NOT NULL,
    schema_version VARCHAR(10) NOT NULL DEFAULT '1.0',
    is_deterministic BOOLEAN NOT NULL DEFAULT TRUE,
    validation_checksum VARCHAR(32),
    
    -- Storage
    storage_path TEXT NOT NULL,
    
    -- Performance metadata
    compilation_duration_ms INTEGER NOT NULL DEFAULT 0,
    estimated_execution_time_ms INTEGER NOT NULL DEFAULT 0,
    memory_footprint_bytes INTEGER NOT NULL DEFAULT 0,
    
    -- Constraints
    CONSTRAINT chk_artifact_type CHECK (artifact_type IN ('execution_plan', 'dependency_graph', 'policy_bundle', 'validation_rules', 'runtime_config')),
    CONSTRAINT chk_optimization_level CHECK (optimization_level IN ('none', 'standard', 'aggressive', 'maximum'))
);

-- Workflow compilations
CREATE TABLE IF NOT EXISTS workflow_compilations (
    compilation_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    workflow_id UUID NOT NULL,
    version_id UUID NOT NULL,
    compilation_target VARCHAR(50) NOT NULL,
    optimization_level VARCHAR(20) NOT NULL,
    compiler_version VARCHAR(20) NOT NULL,
    compiled_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    compilation_duration_ms INTEGER NOT NULL,
    status VARCHAR(20) NOT NULL,
    error_message TEXT,
    artifact_count INTEGER NOT NULL DEFAULT 0,
    total_size_bytes BIGINT NOT NULL DEFAULT 0,
    
    CONSTRAINT chk_compilation_status CHECK (status IN ('pending', 'compiling', 'compiled', 'failed', 'invalid'))
);

-- Indexes
CREATE INDEX IF NOT EXISTS idx_artifacts_workflow ON compiled_artifacts (workflow_id, version_id, artifact_type);
CREATE INDEX IF NOT EXISTS idx_artifacts_compiled_at ON compiled_artifacts (compiled_at DESC);
CREATE INDEX IF NOT EXISTS idx_compilations_workflow ON workflow_compilations (workflow_id, version_id);
CREATE INDEX IF NOT EXISTS idx_compilations_status ON workflow_compilations (status, compiled_at DESC);
"""
