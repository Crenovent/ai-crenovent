"""
Task 7.3-T05: Implement artifact storage (DSL, plan, docs, SBOM)
Centralized artifact repository with checksums and integrity verification
"""

import asyncio
import hashlib
import json
import mimetypes
import os
import tempfile
import uuid
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Union, BinaryIO, Any
from dataclasses import dataclass, field

import aiofiles
from azure.storage.blob.aio import BlobServiceClient
from azure.core.exceptions import ResourceExistsError, ResourceNotFoundError

from ..schemas.canonical_trace_schema import WorkflowTrace


class ArtifactType(Enum):
    """Types of artifacts stored in the registry"""
    DSL = "dsl"
    COMPILED_PLAN = "compiled_plan"
    DOCUMENTATION = "documentation"
    SBOM = "sbom"
    SIGNATURE = "signature"
    ATTESTATION = "attestation"
    METADATA = "metadata"


class StorageStatus(Enum):
    """Storage status of artifacts"""
    UPLOADING = "uploading"
    AVAILABLE = "available"
    ARCHIVED = "archived"
    DELETED = "deleted"


class EncryptionStatus(Enum):
    """Encryption status of artifacts"""
    NONE = "none"
    AT_REST = "at_rest"
    IN_TRANSIT = "in_transit"
    END_TO_END = "end_to_end"


@dataclass
class ArtifactMetadata:
    """Metadata for stored artifacts"""
    artifact_id: str
    version_id: str
    artifact_name: str
    artifact_type: ArtifactType
    content_type: str
    file_size_bytes: int
    content_hash: str
    checksum_algorithm: str = "sha256"
    encoding: str = "utf-8"
    storage_path: str = ""
    storage_bucket: str = ""
    is_signed: bool = False
    signature_path: Optional[str] = None
    encryption_status: EncryptionStatus = EncryptionStatus.NONE
    description: Optional[str] = None
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    uploaded_at: Optional[datetime] = None
    last_accessed_at: Optional[datetime] = None
    access_count: int = 0
    status: StorageStatus = StorageStatus.UPLOADING
    retention_until: Optional[datetime] = None
    tags: Dict[str, str] = field(default_factory=dict)


@dataclass
class StorageConfig:
    """Configuration for artifact storage"""
    storage_type: str = "azure_blob"  # azure_blob, s3, local
    connection_string: Optional[str] = None
    container_name: str = "workflow-registry"
    base_path: str = "artifacts"
    enable_encryption: bool = True
    enable_compression: bool = False
    max_file_size_mb: int = 100
    allowed_extensions: List[str] = field(default_factory=lambda: [
        '.yaml', '.yml', '.json', '.md', '.txt', '.xml', '.pdf', '.zip', '.tar.gz'
    ])
    retention_days_default: int = 2555  # ~7 years
    enable_versioning: bool = True
    enable_soft_delete: bool = True


class ArtifactStorageError(Exception):
    """Base exception for artifact storage operations"""
    pass


class ArtifactNotFoundError(ArtifactStorageError):
    """Artifact not found in storage"""
    pass


class ArtifactIntegrityError(ArtifactStorageError):
    """Artifact integrity check failed"""
    pass


class ArtifactStorageQuotaError(ArtifactStorageError):
    """Storage quota exceeded"""
    pass


class ArtifactStorage:
    """
    Centralized artifact storage with integrity verification and multi-backend support
    """
    
    def __init__(self, config: StorageConfig):
        self.config = config
        self._blob_client = None
        self._local_storage_path = None
        
        if config.storage_type == "azure_blob":
            if not config.connection_string:
                raise ValueError("Azure Blob Storage requires connection_string")
            self._blob_client = BlobServiceClient.from_connection_string(
                config.connection_string
            )
        elif config.storage_type == "local":
            self._local_storage_path = Path(config.base_path)
            self._local_storage_path.mkdir(parents=True, exist_ok=True)
    
    async def store_artifact(
        self,
        content: Union[bytes, str, BinaryIO],
        metadata: ArtifactMetadata,
        overwrite: bool = False
    ) -> ArtifactMetadata:
        """
        Store an artifact with integrity verification
        
        Args:
            content: Artifact content (bytes, string, or file-like object)
            metadata: Artifact metadata
            overwrite: Whether to overwrite existing artifacts
            
        Returns:
            Updated metadata with storage information
            
        Raises:
            ArtifactStorageError: If storage operation fails
        """
        try:
            # Validate content and metadata
            await self._validate_artifact(content, metadata)
            
            # Convert content to bytes if needed
            content_bytes = await self._normalize_content(content)
            
            # Calculate and verify content hash
            calculated_hash = self._calculate_hash(content_bytes, metadata.checksum_algorithm)
            if metadata.content_hash and metadata.content_hash != calculated_hash:
                raise ArtifactIntegrityError(
                    f"Content hash mismatch: expected {metadata.content_hash}, got {calculated_hash}"
                )
            
            # Update metadata with calculated values
            metadata.content_hash = calculated_hash
            metadata.file_size_bytes = len(content_bytes)
            metadata.storage_path = self._generate_storage_path(metadata)
            metadata.storage_bucket = self.config.container_name
            
            # Store the artifact
            if self.config.storage_type == "azure_blob":
                await self._store_azure_blob(content_bytes, metadata, overwrite)
            elif self.config.storage_type == "local":
                await self._store_local_file(content_bytes, metadata, overwrite)
            else:
                raise ArtifactStorageError(f"Unsupported storage type: {self.config.storage_type}")
            
            # Update metadata
            metadata.uploaded_at = datetime.now(timezone.utc)
            metadata.status = StorageStatus.AVAILABLE
            
            return metadata
            
        except Exception as e:
            metadata.status = StorageStatus.DELETED
            raise ArtifactStorageError(f"Failed to store artifact: {str(e)}") from e
    
    async def retrieve_artifact(
        self,
        artifact_id: str,
        verify_integrity: bool = True
    ) -> tuple[bytes, ArtifactMetadata]:
        """
        Retrieve an artifact with optional integrity verification
        
        Args:
            artifact_id: Unique artifact identifier
            verify_integrity: Whether to verify content integrity
            
        Returns:
            Tuple of (content_bytes, metadata)
            
        Raises:
            ArtifactNotFoundError: If artifact doesn't exist
            ArtifactIntegrityError: If integrity check fails
        """
        try:
            # Get metadata (would typically come from database)
            metadata = await self._get_artifact_metadata(artifact_id)
            
            if metadata.status == StorageStatus.DELETED:
                raise ArtifactNotFoundError(f"Artifact {artifact_id} has been deleted")
            
            # Retrieve content
            if self.config.storage_type == "azure_blob":
                content_bytes = await self._retrieve_azure_blob(metadata)
            elif self.config.storage_type == "local":
                content_bytes = await self._retrieve_local_file(metadata)
            else:
                raise ArtifactStorageError(f"Unsupported storage type: {self.config.storage_type}")
            
            # Verify integrity if requested
            if verify_integrity:
                calculated_hash = self._calculate_hash(content_bytes, metadata.checksum_algorithm)
                if calculated_hash != metadata.content_hash:
                    raise ArtifactIntegrityError(
                        f"Integrity check failed for artifact {artifact_id}: "
                        f"expected {metadata.content_hash}, got {calculated_hash}"
                    )
            
            # Update access tracking
            await self._update_access_tracking(metadata)
            
            return content_bytes, metadata
            
        except (ResourceNotFoundError, FileNotFoundError) as e:
            raise ArtifactNotFoundError(f"Artifact {artifact_id} not found") from e
    
    async def delete_artifact(
        self,
        artifact_id: str,
        soft_delete: bool = None
    ) -> bool:
        """
        Delete an artifact (soft or hard delete)
        
        Args:
            artifact_id: Unique artifact identifier
            soft_delete: Whether to soft delete (None uses config default)
            
        Returns:
            True if deletion was successful
        """
        if soft_delete is None:
            soft_delete = self.config.enable_soft_delete
        
        try:
            metadata = await self._get_artifact_metadata(artifact_id)
            
            if soft_delete:
                # Soft delete - mark as deleted but keep the file
                metadata.status = StorageStatus.DELETED
                await self._update_artifact_metadata(metadata)
            else:
                # Hard delete - remove the file
                if self.config.storage_type == "azure_blob":
                    await self._delete_azure_blob(metadata)
                elif self.config.storage_type == "local":
                    await self._delete_local_file(metadata)
                
                metadata.status = StorageStatus.DELETED
                await self._update_artifact_metadata(metadata)
            
            return True
            
        except Exception as e:
            raise ArtifactStorageError(f"Failed to delete artifact {artifact_id}: {str(e)}") from e
    
    async def list_artifacts(
        self,
        version_id: Optional[str] = None,
        artifact_type: Optional[ArtifactType] = None,
        status: Optional[StorageStatus] = None,
        limit: int = 100,
        offset: int = 0
    ) -> List[ArtifactMetadata]:
        """
        List artifacts with optional filtering
        
        Args:
            version_id: Filter by version ID
            artifact_type: Filter by artifact type
            status: Filter by status
            limit: Maximum number of results
            offset: Offset for pagination
            
        Returns:
            List of artifact metadata
        """
        # This would typically query the database
        # For now, return empty list as placeholder
        return []
    
    async def verify_integrity(
        self,
        artifact_id: str
    ) -> Dict[str, Any]:
        """
        Verify the integrity of a stored artifact
        
        Args:
            artifact_id: Unique artifact identifier
            
        Returns:
            Dictionary with integrity check results
        """
        try:
            content_bytes, metadata = await self.retrieve_artifact(artifact_id, verify_integrity=False)
            
            # Calculate current hash
            calculated_hash = self._calculate_hash(content_bytes, metadata.checksum_algorithm)
            
            # Check integrity
            is_valid = calculated_hash == metadata.content_hash
            
            return {
                "artifact_id": artifact_id,
                "is_valid": is_valid,
                "stored_hash": metadata.content_hash,
                "calculated_hash": calculated_hash,
                "algorithm": metadata.checksum_algorithm,
                "file_size": len(content_bytes),
                "stored_size": metadata.file_size_bytes,
                "size_match": len(content_bytes) == metadata.file_size_bytes,
                "verified_at": datetime.now(timezone.utc).isoformat()
            }
            
        except Exception as e:
            return {
                "artifact_id": artifact_id,
                "is_valid": False,
                "error": str(e),
                "verified_at": datetime.now(timezone.utc).isoformat()
            }
    
    async def get_storage_stats(self) -> Dict[str, Any]:
        """
        Get storage statistics
        
        Returns:
            Dictionary with storage statistics
        """
        # This would typically aggregate from database and storage backend
        return {
            "total_artifacts": 0,
            "total_size_bytes": 0,
            "artifacts_by_type": {},
            "artifacts_by_status": {},
            "storage_utilization": 0.0,
            "last_updated": datetime.now(timezone.utc).isoformat()
        }
    
    # Private methods
    
    async def _validate_artifact(self, content: Any, metadata: ArtifactMetadata) -> None:
        """Validate artifact content and metadata"""
        # Check file size limits
        if hasattr(content, '__len__'):
            size_mb = len(content) / (1024 * 1024)
            if size_mb > self.config.max_file_size_mb:
                raise ArtifactStorageError(
                    f"File size {size_mb:.2f}MB exceeds limit of {self.config.max_file_size_mb}MB"
                )
        
        # Check file extension if applicable
        if metadata.artifact_name:
            ext = Path(metadata.artifact_name).suffix.lower()
            if ext and ext not in self.config.allowed_extensions:
                raise ArtifactStorageError(f"File extension {ext} not allowed")
        
        # Validate artifact type
        if not isinstance(metadata.artifact_type, ArtifactType):
            raise ArtifactStorageError("Invalid artifact type")
    
    async def _normalize_content(self, content: Union[bytes, str, BinaryIO]) -> bytes:
        """Convert content to bytes"""
        if isinstance(content, bytes):
            return content
        elif isinstance(content, str):
            return content.encode('utf-8')
        elif hasattr(content, 'read'):
            # File-like object
            if hasattr(content, 'seek'):
                content.seek(0)
            return content.read()
        else:
            raise ArtifactStorageError(f"Unsupported content type: {type(content)}")
    
    def _calculate_hash(self, content: bytes, algorithm: str = "sha256") -> str:
        """Calculate hash of content"""
        if algorithm == "sha256":
            return hashlib.sha256(content).hexdigest()
        elif algorithm == "sha512":
            return hashlib.sha512(content).hexdigest()
        elif algorithm == "md5":
            return hashlib.md5(content).hexdigest()
        else:
            raise ArtifactStorageError(f"Unsupported hash algorithm: {algorithm}")
    
    def _generate_storage_path(self, metadata: ArtifactMetadata) -> str:
        """Generate storage path for artifact"""
        # Create hierarchical path: artifacts/{version_id}/{artifact_type}/{artifact_id}
        return f"{self.config.base_path}/{metadata.version_id}/{metadata.artifact_type.value}/{metadata.artifact_id}"
    
    async def _store_azure_blob(
        self,
        content: bytes,
        metadata: ArtifactMetadata,
        overwrite: bool
    ) -> None:
        """Store artifact in Azure Blob Storage"""
        try:
            # Get container client
            container_client = self._blob_client.get_container_client(self.config.container_name)
            
            # Create container if it doesn't exist
            try:
                await container_client.create_container()
            except ResourceExistsError:
                pass
            
            # Get blob client
            blob_client = container_client.get_blob_client(metadata.storage_path)
            
            # Check if blob exists and overwrite is False
            if not overwrite:
                try:
                    await blob_client.get_blob_properties()
                    raise ArtifactStorageError(f"Artifact already exists: {metadata.artifact_id}")
                except ResourceNotFoundError:
                    pass
            
            # Upload blob with metadata
            blob_metadata = {
                "artifact_id": metadata.artifact_id,
                "version_id": metadata.version_id,
                "artifact_type": metadata.artifact_type.value,
                "content_hash": metadata.content_hash,
                "checksum_algorithm": metadata.checksum_algorithm
            }
            
            await blob_client.upload_blob(
                content,
                content_type=metadata.content_type,
                metadata=blob_metadata,
                overwrite=overwrite
            )
            
        except Exception as e:
            raise ArtifactStorageError(f"Failed to store in Azure Blob: {str(e)}") from e
    
    async def _store_local_file(
        self,
        content: bytes,
        metadata: ArtifactMetadata,
        overwrite: bool
    ) -> None:
        """Store artifact in local filesystem"""
        try:
            file_path = self._local_storage_path / metadata.storage_path
            
            # Create directory structure
            file_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Check if file exists and overwrite is False
            if file_path.exists() and not overwrite:
                raise ArtifactStorageError(f"Artifact already exists: {metadata.artifact_id}")
            
            # Write file
            async with aiofiles.open(file_path, 'wb') as f:
                await f.write(content)
            
            # Write metadata file
            metadata_path = file_path.with_suffix(file_path.suffix + '.meta')
            async with aiofiles.open(metadata_path, 'w') as f:
                await f.write(json.dumps({
                    "artifact_id": metadata.artifact_id,
                    "version_id": metadata.version_id,
                    "artifact_type": metadata.artifact_type.value,
                    "content_hash": metadata.content_hash,
                    "checksum_algorithm": metadata.checksum_algorithm,
                    "created_at": metadata.created_at.isoformat()
                }, indent=2))
                
        except Exception as e:
            raise ArtifactStorageError(f"Failed to store local file: {str(e)}") from e
    
    async def _retrieve_azure_blob(self, metadata: ArtifactMetadata) -> bytes:
        """Retrieve artifact from Azure Blob Storage"""
        try:
            blob_client = self._blob_client.get_blob_client(
                container=self.config.container_name,
                blob=metadata.storage_path
            )
            
            download_stream = await blob_client.download_blob()
            return await download_stream.readall()
            
        except Exception as e:
            raise ArtifactStorageError(f"Failed to retrieve from Azure Blob: {str(e)}") from e
    
    async def _retrieve_local_file(self, metadata: ArtifactMetadata) -> bytes:
        """Retrieve artifact from local filesystem"""
        try:
            file_path = self._local_storage_path / metadata.storage_path
            
            if not file_path.exists():
                raise FileNotFoundError(f"File not found: {file_path}")
            
            async with aiofiles.open(file_path, 'rb') as f:
                return await f.read()
                
        except Exception as e:
            raise ArtifactStorageError(f"Failed to retrieve local file: {str(e)}") from e
    
    async def _delete_azure_blob(self, metadata: ArtifactMetadata) -> None:
        """Delete artifact from Azure Blob Storage"""
        try:
            blob_client = self._blob_client.get_blob_client(
                container=self.config.container_name,
                blob=metadata.storage_path
            )
            
            await blob_client.delete_blob()
            
        except Exception as e:
            raise ArtifactStorageError(f"Failed to delete from Azure Blob: {str(e)}") from e
    
    async def _delete_local_file(self, metadata: ArtifactMetadata) -> None:
        """Delete artifact from local filesystem"""
        try:
            file_path = self._local_storage_path / metadata.storage_path
            metadata_path = file_path.with_suffix(file_path.suffix + '.meta')
            
            if file_path.exists():
                file_path.unlink()
            
            if metadata_path.exists():
                metadata_path.unlink()
                
        except Exception as e:
            raise ArtifactStorageError(f"Failed to delete local file: {str(e)}") from e
    
    async def _get_artifact_metadata(self, artifact_id: str) -> ArtifactMetadata:
        """Get artifact metadata (placeholder - would query database)"""
        # This would typically query the database
        # For now, return a placeholder
        raise ArtifactNotFoundError(f"Artifact metadata not found: {artifact_id}")
    
    async def _update_artifact_metadata(self, metadata: ArtifactMetadata) -> None:
        """Update artifact metadata (placeholder - would update database)"""
        # This would typically update the database
        pass
    
    async def _update_access_tracking(self, metadata: ArtifactMetadata) -> None:
        """Update access tracking for artifact"""
        metadata.last_accessed_at = datetime.now(timezone.utc)
        metadata.access_count += 1
        await self._update_artifact_metadata(metadata)


class WorkflowArtifactManager:
    """
    High-level manager for workflow artifacts with business logic
    """
    
    def __init__(self, storage: ArtifactStorage):
        self.storage = storage
    
    async def store_dsl_workflow(
        self,
        version_id: str,
        dsl_content: str,
        workflow_name: str,
        description: Optional[str] = None
    ) -> ArtifactMetadata:
        """Store DSL workflow content"""
        artifact_id = str(uuid.uuid4())
        
        metadata = ArtifactMetadata(
            artifact_id=artifact_id,
            version_id=version_id,
            artifact_name=f"{workflow_name}.yaml",
            artifact_type=ArtifactType.DSL,
            content_type="application/x-yaml",
            file_size_bytes=0,  # Will be calculated
            content_hash="",    # Will be calculated
            description=description or f"DSL content for {workflow_name}"
        )
        
        return await self.storage.store_artifact(dsl_content, metadata)
    
    async def store_compiled_plan(
        self,
        version_id: str,
        plan_content: Dict[str, Any],
        workflow_name: str
    ) -> ArtifactMetadata:
        """Store compiled execution plan"""
        artifact_id = str(uuid.uuid4())
        
        metadata = ArtifactMetadata(
            artifact_id=artifact_id,
            version_id=version_id,
            artifact_name=f"{workflow_name}_compiled.json",
            artifact_type=ArtifactType.COMPILED_PLAN,
            content_type="application/json",
            file_size_bytes=0,
            content_hash="",
            description=f"Compiled execution plan for {workflow_name}"
        )
        
        plan_json = json.dumps(plan_content, indent=2)
        return await self.storage.store_artifact(plan_json, metadata)
    
    async def store_documentation(
        self,
        version_id: str,
        doc_content: str,
        doc_name: str,
        doc_format: str = "markdown"
    ) -> ArtifactMetadata:
        """Store workflow documentation"""
        artifact_id = str(uuid.uuid4())
        
        content_types = {
            "markdown": "text/markdown",
            "html": "text/html",
            "pdf": "application/pdf",
            "txt": "text/plain"
        }
        
        extensions = {
            "markdown": ".md",
            "html": ".html",
            "pdf": ".pdf",
            "txt": ".txt"
        }
        
        metadata = ArtifactMetadata(
            artifact_id=artifact_id,
            version_id=version_id,
            artifact_name=f"{doc_name}{extensions.get(doc_format, '.txt')}",
            artifact_type=ArtifactType.DOCUMENTATION,
            content_type=content_types.get(doc_format, "text/plain"),
            file_size_bytes=0,
            content_hash="",
            description=f"Documentation for {doc_name}"
        )
        
        return await self.storage.store_artifact(doc_content, metadata)
    
    async def store_sbom(
        self,
        version_id: str,
        sbom_data: Dict[str, Any],
        workflow_name: str,
        sbom_format: str = "CycloneDX"
    ) -> ArtifactMetadata:
        """Store Software Bill of Materials"""
        artifact_id = str(uuid.uuid4())
        
        metadata = ArtifactMetadata(
            artifact_id=artifact_id,
            version_id=version_id,
            artifact_name=f"{workflow_name}_sbom.json",
            artifact_type=ArtifactType.SBOM,
            content_type="application/json",
            file_size_bytes=0,
            content_hash="",
            description=f"SBOM ({sbom_format}) for {workflow_name}",
            tags={"sbom_format": sbom_format}
        )
        
        sbom_json = json.dumps(sbom_data, indent=2)
        return await self.storage.store_artifact(sbom_json, metadata)
    
    async def get_workflow_artifacts(
        self,
        version_id: str
    ) -> Dict[ArtifactType, ArtifactMetadata]:
        """Get all artifacts for a workflow version"""
        artifacts = await self.storage.list_artifacts(version_id=version_id)
        return {artifact.artifact_type: artifact for artifact in artifacts}
    
    async def verify_workflow_integrity(
        self,
        version_id: str
    ) -> Dict[str, Any]:
        """Verify integrity of all artifacts for a workflow version"""
        artifacts = await self.storage.list_artifacts(version_id=version_id)
        
        results = {
            "version_id": version_id,
            "total_artifacts": len(artifacts),
            "verified_artifacts": 0,
            "failed_artifacts": 0,
            "artifact_results": {},
            "overall_valid": True,
            "verified_at": datetime.now(timezone.utc).isoformat()
        }
        
        for artifact in artifacts:
            integrity_result = await self.storage.verify_integrity(artifact.artifact_id)
            results["artifact_results"][artifact.artifact_id] = integrity_result
            
            if integrity_result["is_valid"]:
                results["verified_artifacts"] += 1
            else:
                results["failed_artifacts"] += 1
                results["overall_valid"] = False
        
        return results


# Example usage and testing
async def example_usage():
    """Example usage of the artifact storage system"""
    
    # Configure storage
    config = StorageConfig(
        storage_type="local",  # or "azure_blob"
        base_path="./test_artifacts",
        max_file_size_mb=50
    )
    
    # Initialize storage
    storage = ArtifactStorage(config)
    manager = WorkflowArtifactManager(storage)
    
    # Store a DSL workflow
    version_id = str(uuid.uuid4())
    
    dsl_content = """
    workflow:
      name: "example_pipeline_hygiene"
      version: "1.0.0"
      steps:
        - name: "validate_opportunities"
          type: "query"
          params:
            table: "opportunities"
            filters:
              - "stage != 'Closed Won'"
              - "close_date < NOW()"
    """
    
    try:
        # Store DSL
        dsl_metadata = await manager.store_dsl_workflow(
            version_id=version_id,
            dsl_content=dsl_content,
            workflow_name="example_pipeline_hygiene",
            description="Example pipeline hygiene workflow"
        )
        print(f"Stored DSL artifact: {dsl_metadata.artifact_id}")
        
        # Store compiled plan
        compiled_plan = {
            "workflow_id": str(uuid.uuid4()),
            "version": "1.0.0",
            "execution_graph": {
                "nodes": [
                    {"id": "validate_opportunities", "type": "query"}
                ],
                "edges": []
            }
        }
        
        plan_metadata = await manager.store_compiled_plan(
            version_id=version_id,
            plan_content=compiled_plan,
            workflow_name="example_pipeline_hygiene"
        )
        print(f"Stored compiled plan: {plan_metadata.artifact_id}")
        
        # Verify integrity
        integrity_results = await manager.verify_workflow_integrity(version_id)
        print(f"Integrity check: {integrity_results['overall_valid']}")
        
    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    asyncio.run(example_usage())
