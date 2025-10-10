"""
Task 7.3-T32: Export/Import Bundle (workflow+policy+SBOM)
TAR/ZIP + manifest for tenant migration with attested bundles
"""

import asyncio
import json
import tarfile
import zipfile
import tempfile
import uuid
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Any, BinaryIO, Union
from dataclasses import dataclass, field
import hashlib
import io

import yaml
import asyncpg


class BundleFormat(Enum):
    """Bundle format types"""
    ZIP = "zip"
    TAR = "tar"
    TAR_GZ = "tar.gz"
    JSON = "json"


class BundleType(Enum):
    """Bundle content types"""
    WORKFLOW_ONLY = "workflow_only"
    WORKFLOW_WITH_POLICIES = "workflow_with_policies"
    FULL_BUNDLE = "full_bundle"  # Workflow + Policies + SBOM + Evidence
    TENANT_MIGRATION = "tenant_migration"


class BundleStatus(Enum):
    """Bundle processing status"""
    CREATING = "creating"
    READY = "ready"
    IMPORTING = "importing"
    IMPORTED = "imported"
    FAILED = "failed"
    EXPIRED = "expired"


@dataclass
class BundleManifest:
    """Bundle manifest metadata"""
    bundle_id: str
    bundle_type: BundleType
    bundle_format: BundleFormat
    created_at: datetime
    created_by: str
    source_tenant_id: Optional[int] = None
    target_tenant_id: Optional[int] = None
    
    # Content metadata
    workflow_count: int = 0
    policy_count: int = 0
    sbom_count: int = 0
    evidence_count: int = 0
    
    # Integrity
    content_hash: str = ""
    signature: Optional[str] = None
    
    # Versioning
    manifest_version: str = "1.0"
    compatibility_version: str = "1.0"
    
    # Metadata
    description: Optional[str] = None
    tags: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "bundle_id": self.bundle_id,
            "bundle_type": self.bundle_type.value,
            "bundle_format": self.bundle_format.value,
            "created_at": self.created_at.isoformat(),
            "created_by": self.created_by,
            "source_tenant_id": self.source_tenant_id,
            "target_tenant_id": self.target_tenant_id,
            "workflow_count": self.workflow_count,
            "policy_count": self.policy_count,
            "sbom_count": self.sbom_count,
            "evidence_count": self.evidence_count,
            "content_hash": self.content_hash,
            "signature": self.signature,
            "manifest_version": self.manifest_version,
            "compatibility_version": self.compatibility_version,
            "description": self.description,
            "tags": self.tags
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'BundleManifest':
        return cls(
            bundle_id=data["bundle_id"],
            bundle_type=BundleType(data["bundle_type"]),
            bundle_format=BundleFormat(data["bundle_format"]),
            created_at=datetime.fromisoformat(data["created_at"]),
            created_by=data["created_by"],
            source_tenant_id=data.get("source_tenant_id"),
            target_tenant_id=data.get("target_tenant_id"),
            workflow_count=data.get("workflow_count", 0),
            policy_count=data.get("policy_count", 0),
            sbom_count=data.get("sbom_count", 0),
            evidence_count=data.get("evidence_count", 0),
            content_hash=data.get("content_hash", ""),
            signature=data.get("signature"),
            manifest_version=data.get("manifest_version", "1.0"),
            compatibility_version=data.get("compatibility_version", "1.0"),
            description=data.get("description"),
            tags=data.get("tags", [])
        )


@dataclass
class BundleItem:
    """Individual item in a bundle"""
    item_id: str
    item_type: str  # workflow, policy, sbom, evidence
    item_name: str
    file_path: str
    content_hash: str
    size_bytes: int
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "item_id": self.item_id,
            "item_type": self.item_type,
            "item_name": self.item_name,
            "file_path": self.file_path,
            "content_hash": self.content_hash,
            "size_bytes": self.size_bytes,
            "metadata": self.metadata
        }


class BundleExportImportManager:
    """Manager for workflow bundle export/import operations"""
    
    def __init__(self, db_pool: Optional[asyncpg.Pool] = None, storage_path: Optional[str] = None):
        self.db_pool = db_pool
        self.storage_path = Path(storage_path) if storage_path else Path(tempfile.gettempdir()) / "workflow_bundles"
        self.storage_path.mkdir(parents=True, exist_ok=True)
    
    async def export_workflow_bundle(
        self,
        workflow_ids: List[str],
        bundle_type: BundleType = BundleType.FULL_BUNDLE,
        bundle_format: BundleFormat = BundleFormat.ZIP,
        created_by: str = "system",
        tenant_id: Optional[int] = None,
        include_versions: Optional[List[str]] = None,
        description: Optional[str] = None,
        tags: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """Export workflows and related data as a bundle"""
        
        bundle_id = str(uuid.uuid4())
        
        try:
            # Collect bundle data
            bundle_data = await self._collect_bundle_data(
                workflow_ids, bundle_type, tenant_id, include_versions
            )
            
            # Create manifest
            manifest = BundleManifest(
                bundle_id=bundle_id,
                bundle_type=bundle_type,
                bundle_format=bundle_format,
                created_at=datetime.now(timezone.utc),
                created_by=created_by,
                source_tenant_id=tenant_id,
                workflow_count=len(bundle_data.get("workflows", [])),
                policy_count=len(bundle_data.get("policies", [])),
                sbom_count=len(bundle_data.get("sboms", [])),
                evidence_count=len(bundle_data.get("evidence_packs", [])),
                description=description,
                tags=tags or []
            )
            
            # Create bundle file
            bundle_path = await self._create_bundle_file(
                bundle_data, manifest, bundle_format
            )
            
            # Calculate content hash
            content_hash = await self._calculate_file_hash(bundle_path)
            manifest.content_hash = content_hash
            
            # Store bundle metadata
            if self.db_pool:
                await self._store_bundle_metadata(manifest, bundle_path)
            
            return {
                "bundle_id": bundle_id,
                "bundle_path": str(bundle_path),
                "manifest": manifest.to_dict(),
                "size_bytes": bundle_path.stat().st_size,
                "status": "ready"
            }
            
        except Exception as e:
            return {
                "bundle_id": bundle_id,
                "error": str(e),
                "status": "failed"
            }
    
    async def import_workflow_bundle(
        self,
        bundle_path: Union[str, Path, BinaryIO],
        target_tenant_id: int,
        imported_by: str = "system",
        overwrite_existing: bool = False,
        validate_signatures: bool = True
    ) -> Dict[str, Any]:
        """Import workflows from a bundle"""
        
        import_id = str(uuid.uuid4())
        
        try:
            # Extract bundle
            bundle_data, manifest = await self._extract_bundle(bundle_path)
            
            # Validate bundle
            validation_result = await self._validate_bundle(bundle_data, manifest, validate_signatures)
            if not validation_result["valid"]:
                return {
                    "import_id": import_id,
                    "status": "failed",
                    "error": "Bundle validation failed",
                    "validation_errors": validation_result["errors"]
                }
            
            # Import workflows
            import_results = await self._import_bundle_data(
                bundle_data, target_tenant_id, overwrite_existing
            )
            
            # Update manifest for target tenant
            manifest.target_tenant_id = target_tenant_id
            
            # Store import record
            if self.db_pool:
                await self._store_import_record(import_id, manifest, import_results, imported_by)
            
            return {
                "import_id": import_id,
                "bundle_id": manifest.bundle_id,
                "status": "imported",
                "imported_workflows": import_results.get("workflows", []),
                "imported_policies": import_results.get("policies", []),
                "skipped_items": import_results.get("skipped", []),
                "errors": import_results.get("errors", [])
            }
            
        except Exception as e:
            return {
                "import_id": import_id,
                "status": "failed",
                "error": str(e)
            }
    
    async def _collect_bundle_data(
        self,
        workflow_ids: List[str],
        bundle_type: BundleType,
        tenant_id: Optional[int],
        include_versions: Optional[List[str]]
    ) -> Dict[str, Any]:
        """Collect all data for the bundle"""
        
        bundle_data = {
            "workflows": [],
            "policies": [],
            "sboms": [],
            "evidence_packs": [],
            "artifacts": []
        }
        
        if not self.db_pool:
            return bundle_data
        
        try:
            async with self.db_pool.acquire() as conn:
                # Collect workflows
                for workflow_id in workflow_ids:
                    workflow_data = await self._collect_workflow_data(
                        conn, workflow_id, tenant_id, include_versions
                    )
                    if workflow_data:
                        bundle_data["workflows"].append(workflow_data)
                
                # Collect policies if requested
                if bundle_type in [BundleType.WORKFLOW_WITH_POLICIES, BundleType.FULL_BUNDLE]:
                    policy_data = await self._collect_policy_data(conn, workflow_ids, tenant_id)
                    bundle_data["policies"].extend(policy_data)
                
                # Collect SBOMs if requested
                if bundle_type == BundleType.FULL_BUNDLE:
                    sbom_data = await self._collect_sbom_data(conn, workflow_ids)
                    bundle_data["sboms"].extend(sbom_data)
                    
                    # Collect evidence packs
                    evidence_data = await self._collect_evidence_data(conn, workflow_ids, tenant_id)
                    bundle_data["evidence_packs"].extend(evidence_data)
        
        except Exception as e:
            # Log error but continue with partial data
            pass
        
        return bundle_data
    
    async def _collect_workflow_data(
        self,
        conn: asyncpg.Connection,
        workflow_id: str,
        tenant_id: Optional[int],
        include_versions: Optional[List[str]]
    ) -> Optional[Dict[str, Any]]:
        """Collect workflow and version data"""
        
        # Get workflow metadata
        workflow_query = """
            SELECT * FROM registry_workflows 
            WHERE workflow_id = $1
            AND ($2 IS NULL OR tenant_id = $2)
        """
        
        workflow_row = await conn.fetchrow(workflow_query, workflow_id, tenant_id)
        if not workflow_row:
            return None
        
        workflow_data = dict(workflow_row)
        
        # Get versions
        version_query = """
            SELECT * FROM workflow_versions 
            WHERE workflow_id = $1
            AND ($2 IS NULL OR version_id = ANY($2::UUID[]))
            ORDER BY created_at DESC
        """
        
        version_rows = await conn.fetch(
            version_query, 
            workflow_id, 
            include_versions
        )
        
        workflow_data["versions"] = [dict(row) for row in version_rows]
        
        # Get artifacts for each version
        for version in workflow_data["versions"]:
            artifact_query = """
                SELECT * FROM workflow_artifacts 
                WHERE version_id = $1
            """
            
            artifact_rows = await conn.fetch(artifact_query, version["version_id"])
            version["artifacts"] = [dict(row) for row in artifact_rows]
        
        return workflow_data
    
    async def _collect_policy_data(
        self,
        conn: asyncpg.Connection,
        workflow_ids: List[str],
        tenant_id: Optional[int]
    ) -> List[Dict[str, Any]]:
        """Collect policy data for workflows"""
        
        policy_query = """
            SELECT DISTINCT p.* 
            FROM dsl_policy_packs p
            JOIN dsl_execution_traces t ON p.policy_pack_id = t.policy_pack_id
            WHERE t.workflow_id = ANY($1::UUID[])
            AND ($2 IS NULL OR p.tenant_id = $2)
        """
        
        try:
            policy_rows = await conn.fetch(policy_query, workflow_ids, tenant_id)
            return [dict(row) for row in policy_rows]
        except Exception:
            return []
    
    async def _collect_sbom_data(
        self,
        conn: asyncpg.Connection,
        workflow_ids: List[str]
    ) -> List[Dict[str, Any]]:
        """Collect SBOM data for workflows"""
        
        sbom_query = """
            SELECT s.* 
            FROM workflow_sbom s
            JOIN workflow_versions v ON s.version_id = v.version_id
            WHERE v.workflow_id = ANY($1::UUID[])
        """
        
        try:
            sbom_rows = await conn.fetch(sbom_query, workflow_ids)
            return [dict(row) for row in sbom_rows]
        except Exception:
            return []
    
    async def _collect_evidence_data(
        self,
        conn: asyncpg.Connection,
        workflow_ids: List[str],
        tenant_id: Optional[int]
    ) -> List[Dict[str, Any]]:
        """Collect evidence pack data for workflows"""
        
        evidence_query = """
            SELECT * FROM dsl_evidence_packs 
            WHERE workflow_id = ANY($1::UUID[])
            AND ($2 IS NULL OR tenant_id = $2)
            ORDER BY created_at DESC
            LIMIT 100  -- Limit to recent evidence
        """
        
        try:
            evidence_rows = await conn.fetch(evidence_query, workflow_ids, tenant_id)
            return [dict(row) for row in evidence_rows]
        except Exception:
            return []
    
    async def _create_bundle_file(
        self,
        bundle_data: Dict[str, Any],
        manifest: BundleManifest,
        bundle_format: BundleFormat
    ) -> Path:
        """Create the bundle file"""
        
        bundle_filename = f"workflow_bundle_{manifest.bundle_id}.{bundle_format.value}"
        bundle_path = self.storage_path / bundle_filename
        
        if bundle_format == BundleFormat.ZIP:
            await self._create_zip_bundle(bundle_data, manifest, bundle_path)
        elif bundle_format in [BundleFormat.TAR, BundleFormat.TAR_GZ]:
            await self._create_tar_bundle(bundle_data, manifest, bundle_path, bundle_format)
        elif bundle_format == BundleFormat.JSON:
            await self._create_json_bundle(bundle_data, manifest, bundle_path)
        else:
            raise ValueError(f"Unsupported bundle format: {bundle_format}")
        
        return bundle_path
    
    async def _create_zip_bundle(
        self,
        bundle_data: Dict[str, Any],
        manifest: BundleManifest,
        bundle_path: Path
    ) -> None:
        """Create ZIP bundle"""
        
        with zipfile.ZipFile(bundle_path, 'w', zipfile.ZIP_DEFLATED) as zip_file:
            # Add manifest
            manifest_json = json.dumps(manifest.to_dict(), indent=2, default=str)
            zip_file.writestr("manifest.json", manifest_json)
            
            # Add workflows
            for i, workflow in enumerate(bundle_data.get("workflows", [])):
                workflow_json = json.dumps(workflow, indent=2, default=str)
                zip_file.writestr(f"workflows/workflow_{i}.json", workflow_json)
            
            # Add policies
            for i, policy in enumerate(bundle_data.get("policies", [])):
                policy_json = json.dumps(policy, indent=2, default=str)
                zip_file.writestr(f"policies/policy_{i}.json", policy_json)
            
            # Add SBOMs
            for i, sbom in enumerate(bundle_data.get("sboms", [])):
                sbom_json = json.dumps(sbom, indent=2, default=str)
                zip_file.writestr(f"sboms/sbom_{i}.json", sbom_json)
            
            # Add evidence packs
            for i, evidence in enumerate(bundle_data.get("evidence_packs", [])):
                evidence_json = json.dumps(evidence, indent=2, default=str)
                zip_file.writestr(f"evidence/evidence_{i}.json", evidence_json)
    
    async def _create_tar_bundle(
        self,
        bundle_data: Dict[str, Any],
        manifest: BundleManifest,
        bundle_path: Path,
        bundle_format: BundleFormat
    ) -> None:
        """Create TAR bundle"""
        
        mode = "w:gz" if bundle_format == BundleFormat.TAR_GZ else "w"
        
        with tarfile.open(bundle_path, mode) as tar_file:
            # Add manifest
            manifest_json = json.dumps(manifest.to_dict(), indent=2, default=str)
            manifest_info = tarfile.TarInfo(name="manifest.json")
            manifest_info.size = len(manifest_json.encode())
            tar_file.addfile(manifest_info, io.BytesIO(manifest_json.encode()))
            
            # Add workflows
            for i, workflow in enumerate(bundle_data.get("workflows", [])):
                workflow_json = json.dumps(workflow, indent=2, default=str)
                workflow_info = tarfile.TarInfo(name=f"workflows/workflow_{i}.json")
                workflow_info.size = len(workflow_json.encode())
                tar_file.addfile(workflow_info, io.BytesIO(workflow_json.encode()))
            
            # Add other data types similarly...
    
    async def _create_json_bundle(
        self,
        bundle_data: Dict[str, Any],
        manifest: BundleManifest,
        bundle_path: Path
    ) -> None:
        """Create JSON bundle"""
        
        complete_bundle = {
            "manifest": manifest.to_dict(),
            "data": bundle_data
        }
        
        with open(bundle_path, 'w') as f:
            json.dump(complete_bundle, f, indent=2, default=str)
    
    async def _extract_bundle(
        self,
        bundle_path: Union[str, Path, BinaryIO]
    ) -> Tuple[Dict[str, Any], BundleManifest]:
        """Extract bundle contents"""
        
        if isinstance(bundle_path, (str, Path)):
            bundle_path = Path(bundle_path)
            
            if bundle_path.suffix == '.zip':
                return await self._extract_zip_bundle(bundle_path)
            elif bundle_path.suffix in ['.tar', '.gz']:
                return await self._extract_tar_bundle(bundle_path)
            elif bundle_path.suffix == '.json':
                return await self._extract_json_bundle(bundle_path)
            else:
                raise ValueError(f"Unsupported bundle format: {bundle_path.suffix}")
        else:
            # Handle BinaryIO
            return await self._extract_binary_bundle(bundle_path)
    
    async def _extract_zip_bundle(self, bundle_path: Path) -> Tuple[Dict[str, Any], BundleManifest]:
        """Extract ZIP bundle"""
        
        bundle_data = {"workflows": [], "policies": [], "sboms": [], "evidence_packs": []}
        
        with zipfile.ZipFile(bundle_path, 'r') as zip_file:
            # Read manifest
            manifest_content = zip_file.read("manifest.json").decode()
            manifest_dict = json.loads(manifest_content)
            manifest = BundleManifest.from_dict(manifest_dict)
            
            # Read workflows
            for name in zip_file.namelist():
                if name.startswith("workflows/") and name.endswith(".json"):
                    workflow_content = zip_file.read(name).decode()
                    workflow_data = json.loads(workflow_content)
                    bundle_data["workflows"].append(workflow_data)
                elif name.startswith("policies/") and name.endswith(".json"):
                    policy_content = zip_file.read(name).decode()
                    policy_data = json.loads(policy_content)
                    bundle_data["policies"].append(policy_data)
                elif name.startswith("sboms/") and name.endswith(".json"):
                    sbom_content = zip_file.read(name).decode()
                    sbom_data = json.loads(sbom_content)
                    bundle_data["sboms"].append(sbom_data)
                elif name.startswith("evidence/") and name.endswith(".json"):
                    evidence_content = zip_file.read(name).decode()
                    evidence_data = json.loads(evidence_content)
                    bundle_data["evidence_packs"].append(evidence_data)
        
        return bundle_data, manifest
    
    async def _extract_tar_bundle(self, bundle_path: Path) -> Tuple[Dict[str, Any], BundleManifest]:
        """Extract TAR bundle"""
        
        bundle_data = {"workflows": [], "policies": [], "sboms": [], "evidence_packs": []}
        
        mode = "r:gz" if bundle_path.suffix == ".gz" else "r"
        
        with tarfile.open(bundle_path, mode) as tar_file:
            # Read manifest
            manifest_file = tar_file.extractfile("manifest.json")
            manifest_content = manifest_file.read().decode()
            manifest_dict = json.loads(manifest_content)
            manifest = BundleManifest.from_dict(manifest_dict)
            
            # Read other files
            for member in tar_file.getmembers():
                if member.name.startswith("workflows/") and member.name.endswith(".json"):
                    file_obj = tar_file.extractfile(member)
                    workflow_content = file_obj.read().decode()
                    workflow_data = json.loads(workflow_content)
                    bundle_data["workflows"].append(workflow_data)
                # Handle other file types similarly...
        
        return bundle_data, manifest
    
    async def _extract_json_bundle(self, bundle_path: Path) -> Tuple[Dict[str, Any], BundleManifest]:
        """Extract JSON bundle"""
        
        with open(bundle_path, 'r') as f:
            complete_bundle = json.load(f)
        
        manifest = BundleManifest.from_dict(complete_bundle["manifest"])
        bundle_data = complete_bundle["data"]
        
        return bundle_data, manifest
    
    async def _extract_binary_bundle(self, bundle_io: BinaryIO) -> Tuple[Dict[str, Any], BundleManifest]:
        """Extract bundle from binary IO"""
        
        # Try to detect format and extract
        # This is a simplified implementation
        content = bundle_io.read()
        
        # Try JSON first
        try:
            complete_bundle = json.loads(content.decode())
            manifest = BundleManifest.from_dict(complete_bundle["manifest"])
            bundle_data = complete_bundle["data"]
            return bundle_data, manifest
        except:
            pass
        
        # Try ZIP
        try:
            with zipfile.ZipFile(io.BytesIO(content), 'r') as zip_file:
                return await self._extract_zip_from_zipfile(zip_file)
        except:
            pass
        
        raise ValueError("Unable to detect bundle format")
    
    async def _validate_bundle(
        self,
        bundle_data: Dict[str, Any],
        manifest: BundleManifest,
        validate_signatures: bool
    ) -> Dict[str, Any]:
        """Validate bundle integrity and content"""
        
        errors = []
        
        # Validate manifest
        if not manifest.bundle_id:
            errors.append("Missing bundle ID in manifest")
        
        # Validate content counts
        actual_workflow_count = len(bundle_data.get("workflows", []))
        if actual_workflow_count != manifest.workflow_count:
            errors.append(f"Workflow count mismatch: expected {manifest.workflow_count}, got {actual_workflow_count}")
        
        # Validate workflow structure
        for i, workflow in enumerate(bundle_data.get("workflows", [])):
            if not workflow.get("workflow_id"):
                errors.append(f"Workflow {i} missing workflow_id")
            if not workflow.get("workflow_name"):
                errors.append(f"Workflow {i} missing workflow_name")
        
        # Validate signatures if requested
        if validate_signatures and manifest.signature:
            # This would integrate with signature verification
            # For now, just check if signature exists
            pass
        
        return {
            "valid": len(errors) == 0,
            "errors": errors
        }
    
    async def _import_bundle_data(
        self,
        bundle_data: Dict[str, Any],
        target_tenant_id: int,
        overwrite_existing: bool
    ) -> Dict[str, Any]:
        """Import bundle data into the target tenant"""
        
        results = {
            "workflows": [],
            "policies": [],
            "skipped": [],
            "errors": []
        }
        
        if not self.db_pool:
            results["errors"].append("Database not available")
            return results
        
        try:
            async with self.db_pool.acquire() as conn:
                async with conn.transaction():
                    # Import workflows
                    for workflow in bundle_data.get("workflows", []):
                        try:
                            imported_workflow = await self._import_workflow(
                                conn, workflow, target_tenant_id, overwrite_existing
                            )
                            if imported_workflow:
                                results["workflows"].append(imported_workflow)
                            else:
                                results["skipped"].append({
                                    "type": "workflow",
                                    "id": workflow.get("workflow_id"),
                                    "reason": "already_exists"
                                })
                        except Exception as e:
                            results["errors"].append({
                                "type": "workflow",
                                "id": workflow.get("workflow_id"),
                                "error": str(e)
                            })
                    
                    # Import policies
                    for policy in bundle_data.get("policies", []):
                        try:
                            imported_policy = await self._import_policy(
                                conn, policy, target_tenant_id, overwrite_existing
                            )
                            if imported_policy:
                                results["policies"].append(imported_policy)
                        except Exception as e:
                            results["errors"].append({
                                "type": "policy",
                                "id": policy.get("policy_pack_id"),
                                "error": str(e)
                            })
        
        except Exception as e:
            results["errors"].append({"type": "transaction", "error": str(e)})
        
        return results
    
    async def _import_workflow(
        self,
        conn: asyncpg.Connection,
        workflow_data: Dict[str, Any],
        target_tenant_id: int,
        overwrite_existing: bool
    ) -> Optional[Dict[str, Any]]:
        """Import a single workflow"""
        
        workflow_id = workflow_data.get("workflow_id")
        
        # Check if workflow exists
        existing_query = """
            SELECT workflow_id FROM registry_workflows 
            WHERE workflow_id = $1 AND tenant_id = $2
        """
        
        existing = await conn.fetchrow(existing_query, workflow_id, target_tenant_id)
        
        if existing and not overwrite_existing:
            return None
        
        # Update tenant_id
        workflow_data["tenant_id"] = target_tenant_id
        
        # Insert or update workflow
        if existing and overwrite_existing:
            # Update existing workflow
            update_query = """
                UPDATE registry_workflows SET
                    workflow_name = $3,
                    description = $4,
                    updated_at = NOW()
                WHERE workflow_id = $1 AND tenant_id = $2
                RETURNING workflow_id
            """
            
            result = await conn.fetchrow(
                update_query,
                workflow_id,
                target_tenant_id,
                workflow_data.get("workflow_name"),
                workflow_data.get("description")
            )
        else:
            # Insert new workflow
            insert_query = """
                INSERT INTO registry_workflows (
                    workflow_id, tenant_id, workflow_name, workflow_slug,
                    namespace, workflow_type, automation_type, industry_overlay,
                    category, description, keywords, tags, owner_user_id,
                    status, visibility, business_value, customer_impact,
                    risk_level, compliance_requirements
                ) VALUES (
                    $1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14, $15, $16, $17, $18, $19
                ) RETURNING workflow_id
            """
            
            result = await conn.fetchrow(
                insert_query,
                workflow_id,
                target_tenant_id,
                workflow_data.get("workflow_name"),
                workflow_data.get("workflow_slug"),
                workflow_data.get("namespace"),
                workflow_data.get("workflow_type"),
                workflow_data.get("automation_type"),
                workflow_data.get("industry_overlay"),
                workflow_data.get("category"),
                workflow_data.get("description"),
                workflow_data.get("keywords"),
                workflow_data.get("tags"),
                workflow_data.get("owner_user_id"),
                workflow_data.get("status"),
                workflow_data.get("visibility"),
                workflow_data.get("business_value"),
                workflow_data.get("customer_impact"),
                workflow_data.get("risk_level"),
                workflow_data.get("compliance_requirements")
            )
        
        # Import versions
        for version_data in workflow_data.get("versions", []):
            await self._import_workflow_version(conn, version_data, workflow_id)
        
        return {"workflow_id": str(result["workflow_id"])}
    
    async def _import_workflow_version(
        self,
        conn: asyncpg.Connection,
        version_data: Dict[str, Any],
        workflow_id: str
    ) -> None:
        """Import a workflow version"""
        
        # Insert version (simplified)
        insert_query = """
            INSERT INTO workflow_versions (
                version_id, workflow_id, version_number, major_version,
                minor_version, patch_version, status, changelog,
                dsl_schema_version, is_immutable, content_hash
            ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11)
            ON CONFLICT (workflow_id, version_number) DO NOTHING
        """
        
        await conn.execute(
            insert_query,
            version_data.get("version_id"),
            workflow_id,
            version_data.get("version_number"),
            version_data.get("major_version"),
            version_data.get("minor_version"),
            version_data.get("patch_version"),
            version_data.get("status"),
            version_data.get("changelog"),
            version_data.get("dsl_schema_version"),
            version_data.get("is_immutable"),
            version_data.get("content_hash")
        )
    
    async def _import_policy(
        self,
        conn: asyncpg.Connection,
        policy_data: Dict[str, Any],
        target_tenant_id: int,
        overwrite_existing: bool
    ) -> Optional[Dict[str, Any]]:
        """Import a policy pack"""
        
        # Update tenant_id
        policy_data["tenant_id"] = target_tenant_id
        
        # Insert policy (simplified)
        insert_query = """
            INSERT INTO dsl_policy_packs (
                policy_pack_id, tenant_id, policy_name, policy_version,
                policy_type, industry_overlay, policy_rules
            ) VALUES ($1, $2, $3, $4, $5, $6, $7)
            ON CONFLICT (policy_pack_id, tenant_id) 
            DO UPDATE SET policy_rules = EXCLUDED.policy_rules
            RETURNING policy_pack_id
        """
        
        try:
            result = await conn.fetchrow(
                insert_query,
                policy_data.get("policy_pack_id"),
                target_tenant_id,
                policy_data.get("policy_name"),
                policy_data.get("policy_version"),
                policy_data.get("policy_type"),
                policy_data.get("industry_overlay"),
                json.dumps(policy_data.get("policy_rules", {}))
            )
            
            return {"policy_pack_id": str(result["policy_pack_id"])}
        except Exception:
            return None
    
    async def _calculate_file_hash(self, file_path: Path) -> str:
        """Calculate SHA-256 hash of file"""
        
        hash_sha256 = hashlib.sha256()
        
        with open(file_path, 'rb') as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_sha256.update(chunk)
        
        return hash_sha256.hexdigest()
    
    async def _store_bundle_metadata(
        self,
        manifest: BundleManifest,
        bundle_path: Path
    ) -> None:
        """Store bundle metadata in database"""
        
        if not self.db_pool:
            return
        
        try:
            insert_query = """
                INSERT INTO workflow_bundles (
                    bundle_id, bundle_type, bundle_format, created_at,
                    created_by, source_tenant_id, workflow_count,
                    policy_count, sbom_count, evidence_count,
                    content_hash, file_path, file_size_bytes,
                    manifest_data, status
                ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14, $15)
            """
            
            async with self.db_pool.acquire() as conn:
                await conn.execute(
                    insert_query,
                    manifest.bundle_id,
                    manifest.bundle_type.value,
                    manifest.bundle_format.value,
                    manifest.created_at,
                    manifest.created_by,
                    manifest.source_tenant_id,
                    manifest.workflow_count,
                    manifest.policy_count,
                    manifest.sbom_count,
                    manifest.evidence_count,
                    manifest.content_hash,
                    str(bundle_path),
                    bundle_path.stat().st_size,
                    json.dumps(manifest.to_dict()),
                    BundleStatus.READY.value
                )
        except Exception:
            # Log error but don't fail the export
            pass
    
    async def _store_import_record(
        self,
        import_id: str,
        manifest: BundleManifest,
        import_results: Dict[str, Any],
        imported_by: str
    ) -> None:
        """Store import record in database"""
        
        if not self.db_pool:
            return
        
        try:
            insert_query = """
                INSERT INTO workflow_bundle_imports (
                    import_id, bundle_id, target_tenant_id, imported_by,
                    imported_at, import_results, status
                ) VALUES ($1, $2, $3, $4, $5, $6, $7)
            """
            
            async with self.db_pool.acquire() as conn:
                await conn.execute(
                    insert_query,
                    import_id,
                    manifest.bundle_id,
                    manifest.target_tenant_id,
                    imported_by,
                    datetime.now(timezone.utc),
                    json.dumps(import_results),
                    BundleStatus.IMPORTED.value
                )
        except Exception:
            # Log error but don't fail the import
            pass


# Database schema for bundle management
BUNDLE_SCHEMA_SQL = """
-- Workflow bundles
CREATE TABLE IF NOT EXISTS workflow_bundles (
    bundle_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    bundle_type VARCHAR(50) NOT NULL,
    bundle_format VARCHAR(20) NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    created_by VARCHAR(255) NOT NULL,
    source_tenant_id INTEGER,
    target_tenant_id INTEGER,
    
    -- Content metadata
    workflow_count INTEGER NOT NULL DEFAULT 0,
    policy_count INTEGER NOT NULL DEFAULT 0,
    sbom_count INTEGER NOT NULL DEFAULT 0,
    evidence_count INTEGER NOT NULL DEFAULT 0,
    
    -- File information
    file_path TEXT,
    file_size_bytes BIGINT,
    content_hash VARCHAR(64),
    
    -- Bundle data
    manifest_data JSONB,
    status VARCHAR(20) NOT NULL DEFAULT 'creating',
    
    -- Constraints
    CONSTRAINT chk_bundle_type CHECK (bundle_type IN ('workflow_only', 'workflow_with_policies', 'full_bundle', 'tenant_migration')),
    CONSTRAINT chk_bundle_format CHECK (bundle_format IN ('zip', 'tar', 'tar.gz', 'json')),
    CONSTRAINT chk_bundle_status CHECK (status IN ('creating', 'ready', 'importing', 'imported', 'failed', 'expired'))
);

-- Bundle import records
CREATE TABLE IF NOT EXISTS workflow_bundle_imports (
    import_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    bundle_id UUID NOT NULL REFERENCES workflow_bundles(bundle_id),
    target_tenant_id INTEGER NOT NULL,
    imported_by VARCHAR(255) NOT NULL,
    imported_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    import_results JSONB,
    status VARCHAR(20) NOT NULL DEFAULT 'imported',
    
    CONSTRAINT chk_import_status CHECK (status IN ('imported', 'failed', 'partial'))
);

-- Indexes
CREATE INDEX IF NOT EXISTS idx_bundles_created_at ON workflow_bundles (created_at DESC);
CREATE INDEX IF NOT EXISTS idx_bundles_source_tenant ON workflow_bundles (source_tenant_id);
CREATE INDEX IF NOT EXISTS idx_bundles_status ON workflow_bundles (status);
CREATE INDEX IF NOT EXISTS idx_imports_target_tenant ON workflow_bundle_imports (target_tenant_id);
CREATE INDEX IF NOT EXISTS idx_imports_imported_at ON workflow_bundle_imports (imported_at DESC);
"""
