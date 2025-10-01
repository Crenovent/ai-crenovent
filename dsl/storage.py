"""
Workflow Storage - Handles workflow storage in database and Azure Blob Storage
"""

import os
import json
import asyncio
from typing import Dict, List, Any, Optional
from datetime import datetime

from .parser import DSLWorkflow, DSLParser
import logging

logger = logging.getLogger(__name__)

class WorkflowStorage:
    """
    Manages workflow storage between database metadata and Azure Blob Storage content
    """
    
    def __init__(self, pool_manager=None):
        self.pool_manager = pool_manager
        self.logger = logging.getLogger(__name__)
        self.parser = DSLParser()
        
        # Azure Blob Storage configuration
        self.blob_connection_string = os.getenv('AZURE_STORAGE_CONNECTION_STRING')
        self.blob_container_name = os.getenv('AZURE_STORAGE_CONTAINER_NAME', 'dsl-workflows')
    
    async def save_workflow(self, workflow: DSLWorkflow, created_by_user_id: int, tenant_id: str) -> bool:
        """
        Save workflow to database and blob storage
        
        Args:
            workflow: DSL workflow to save
            created_by_user_id: User ID creating the workflow
            tenant_id: Tenant ID for isolation
            
        Returns:
            bool: Success status
        """
        try:
            # Convert workflow to YAML
            yaml_content = self.parser.workflow_to_yaml(workflow)
            
            # Upload to blob storage
            blob_url = await self._upload_to_blob(workflow.workflow_id, yaml_content)
            
            # Calculate content hash
            import hashlib
            content_hash = hashlib.sha256(yaml_content.encode('utf-8')).hexdigest()
            
            # Save metadata to database
            success = await self._save_workflow_metadata(
                workflow, blob_url, content_hash, created_by_user_id, tenant_id
            )
            
            if success:
                self.logger.info(f"Workflow saved successfully: {workflow.workflow_id}")
            
            return success
            
        except Exception as e:
            self.logger.error(f"Error saving workflow {workflow.workflow_id}: {e}")
            return False
    
    async def load_workflow(self, workflow_id: str, tenant_id: str) -> Optional[DSLWorkflow]:
        """
        Load workflow from database and blob storage
        
        Args:
            workflow_id: Workflow ID to load
            tenant_id: Tenant ID for isolation
            
        Returns:
            DSLWorkflow or None if not found
        """
        try:
            # Get workflow metadata from database
            metadata = await self._get_workflow_metadata(workflow_id, tenant_id)
            if not metadata:
                return None
            
            # Download YAML content from blob storage
            yaml_content = await self._download_from_blob(metadata['content_blob_url'])
            if not yaml_content:
                return None
            
            # Parse workflow
            workflow = self.parser.parse_yaml(yaml_content)
            
            self.logger.info(f"Workflow loaded successfully: {workflow_id}")
            return workflow
            
        except Exception as e:
            self.logger.error(f"Error loading workflow {workflow_id}: {e}")
            return None
    
    async def list_workflows(self, tenant_id: str, module: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        List workflows for a tenant
        
        Args:
            tenant_id: Tenant ID for isolation
            module: Optional module filter
            
        Returns:
            List of workflow metadata
        """
        try:
            if not self.pool_manager or not self.pool_manager.postgres_pool:
                return []
            
            async with self.pool_manager.postgres_pool.acquire() as conn:
                await conn.execute(f"SET app.current_tenant = '{tenant_id}'")
                
                # Build query
                where_clause = "WHERE tenant_id = $1"
                params = [tenant_id]
                
                if module:
                    where_clause += " AND module = $2"
                    params.append(module)
                
                query = f"""
                    SELECT 
                        workflow_id, name, description, module, automation_type,
                        version, status, industry, sla_tier, tags,
                        execution_count, success_rate, avg_execution_time_ms,
                        created_at, updated_at, created_by_user_id
                    FROM dsl_workflows
                    {where_clause}
                    ORDER BY updated_at DESC
                """
                
                rows = await conn.fetch(query, *params)
                
                workflows = []
                for row in rows:
                    workflows.append({
                        'workflow_id': row['workflow_id'],
                        'name': row['name'],
                        'description': row['description'],
                        'module': row['module'],
                        'automation_type': row['automation_type'],
                        'version': row['version'],
                        'status': row['status'],
                        'industry': row['industry'],
                        'sla_tier': row['sla_tier'],
                        'tags': row['tags'],
                        'execution_count': row['execution_count'],
                        'success_rate': row['success_rate'],
                        'avg_execution_time_ms': row['avg_execution_time_ms'],
                        'created_at': row['created_at'].isoformat() if row['created_at'] else None,
                        'updated_at': row['updated_at'].isoformat() if row['updated_at'] else None,
                        'created_by_user_id': row['created_by_user_id']
                    })
                
                return workflows
                
        except Exception as e:
            self.logger.error(f"Error listing workflows: {e}")
            return []
    
    async def delete_workflow(self, workflow_id: str, tenant_id: str) -> bool:
        """
        Delete workflow from database and blob storage
        
        Args:
            workflow_id: Workflow ID to delete
            tenant_id: Tenant ID for isolation
            
        Returns:
            bool: Success status
        """
        try:
            # Get workflow metadata
            metadata = await self._get_workflow_metadata(workflow_id, tenant_id)
            if not metadata:
                return False
            
            # Delete from blob storage
            await self._delete_from_blob(metadata['content_blob_url'])
            
            # Delete from database
            success = await self._delete_workflow_metadata(workflow_id, tenant_id)
            
            if success:
                self.logger.info(f"Workflow deleted successfully: {workflow_id}")
            
            return success
            
        except Exception as e:
            self.logger.error(f"Error deleting workflow {workflow_id}: {e}")
            return False
    
    async def update_workflow_status(self, workflow_id: str, tenant_id: str, status: str) -> bool:
        """
        Update workflow status
        
        Args:
            workflow_id: Workflow ID
            tenant_id: Tenant ID
            status: New status (draft, approved, deprecated, retired)
            
        Returns:
            bool: Success status
        """
        try:
            if not self.pool_manager or not self.pool_manager.postgres_pool:
                return False
            
            async with self.pool_manager.postgres_pool.acquire() as conn:
                await conn.execute(f"SET app.current_tenant = '{tenant_id}'")
                
                query = """
                    UPDATE dsl_workflows
                    SET status = $1, updated_at = NOW()
                    WHERE workflow_id = $2 AND tenant_id = $3
                """
                
                result = await conn.execute(query, status, workflow_id, tenant_id)
                
                # Check if any row was updated
                return result.split()[-1] == '1'
                
        except Exception as e:
            self.logger.error(f"Error updating workflow status: {e}")
            return False
    
    async def _save_workflow_metadata(
        self,
        workflow: DSLWorkflow,
        blob_url: str,
        content_hash: str,
        created_by_user_id: int,
        tenant_id: str
    ) -> bool:
        """Save workflow metadata to database"""
        try:
            if not self.pool_manager or not self.pool_manager.postgres_pool:
                return False
            
            async with self.pool_manager.postgres_pool.acquire() as conn:
                await conn.execute(f"SET app.current_tenant = '{tenant_id}'")
                
                # Check if workflow already exists
                existing_query = """
                    SELECT workflow_id FROM dsl_workflows
                    WHERE workflow_id = $1 AND tenant_id = $2
                """
                existing = await conn.fetchrow(existing_query, workflow.workflow_id, tenant_id)
                
                if existing:
                    # Update existing workflow
                    query = """
                        UPDATE dsl_workflows
                        SET 
                            name = $1, description = $2, module = $3, automation_type = $4,
                            version = $5, content_blob_url = $6, content_hash = $7,
                            industry = $8, sla_tier = $9, tags = $10,
                            policy_pack_id = $11, compliance_overlays = $12,
                            trust_threshold = $13, updated_at = NOW()
                        WHERE workflow_id = $14 AND tenant_id = $15
                    """
                    
                    await conn.execute(
                        query,
                        workflow.name,
                        workflow.metadata.get('description', ''),
                        workflow.module,
                        workflow.automation_type,
                        workflow.version,
                        blob_url,
                        content_hash,
                        workflow.metadata.get('industry', 'SaaS'),
                        workflow.metadata.get('sla_tier', 'T2'),
                        json.dumps(workflow.metadata.get('tags', [])),
                        workflow.governance.get('policy_pack'),
                        json.dumps(workflow.governance.get('compliance_overlays', [])),
                        workflow.governance.get('trust_threshold', 0.70),
                        workflow.workflow_id,
                        tenant_id
                    )
                else:
                    # Insert new workflow
                    query = """
                        INSERT INTO dsl_workflows (
                            workflow_id, name, description, module, automation_type,
                            version, content_blob_url, content_hash,
                            industry, sla_tier, tags,
                            policy_pack_id, compliance_overlays, trust_threshold,
                            tenant_id, created_by_user_id
                        ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14, $15, $16)
                    """
                    
                    await conn.execute(
                        query,
                        workflow.workflow_id,
                        workflow.name,
                        workflow.metadata.get('description', ''),
                        workflow.module,
                        workflow.automation_type,
                        workflow.version,
                        blob_url,
                        content_hash,
                        workflow.metadata.get('industry', 'SaaS'),
                        workflow.metadata.get('sla_tier', 'T2'),
                        json.dumps(workflow.metadata.get('tags', [])),
                        workflow.governance.get('policy_pack'),
                        json.dumps(workflow.governance.get('compliance_overlays', [])),
                        workflow.governance.get('trust_threshold', 0.70),
                        tenant_id,
                        created_by_user_id
                    )
                
                return True
                
        except Exception as e:
            self.logger.error(f"Error saving workflow metadata: {e}")
            return False
    
    async def _get_workflow_metadata(self, workflow_id: str, tenant_id: str) -> Optional[Dict[str, Any]]:
        """Get workflow metadata from database"""
        try:
            if not self.pool_manager or not self.pool_manager.postgres_pool:
                return None
            
            async with self.pool_manager.postgres_pool.acquire() as conn:
                await conn.execute(f"SET app.current_tenant = '{tenant_id}'")
                
                query = """
                    SELECT 
                        workflow_id, name, description, module, automation_type,
                        version, status, content_blob_url, content_hash,
                        industry, sla_tier, tags, policy_pack_id,
                        compliance_overlays, trust_threshold,
                        execution_count, success_rate, avg_execution_time_ms,
                        created_at, updated_at, created_by_user_id
                    FROM dsl_workflows
                    WHERE workflow_id = $1 AND tenant_id = $2
                """
                
                row = await conn.fetchrow(query, workflow_id, tenant_id)
                
                if row:
                    return {
                        'workflow_id': row['workflow_id'],
                        'name': row['name'],
                        'description': row['description'],
                        'module': row['module'],
                        'automation_type': row['automation_type'],
                        'version': row['version'],
                        'status': row['status'],
                        'content_blob_url': row['content_blob_url'],
                        'content_hash': row['content_hash'],
                        'industry': row['industry'],
                        'sla_tier': row['sla_tier'],
                        'tags': row['tags'],
                        'policy_pack_id': row['policy_pack_id'],
                        'compliance_overlays': row['compliance_overlays'],
                        'trust_threshold': row['trust_threshold'],
                        'execution_count': row['execution_count'],
                        'success_rate': row['success_rate'],
                        'avg_execution_time_ms': row['avg_execution_time_ms'],
                        'created_at': row['created_at'],
                        'updated_at': row['updated_at'],
                        'created_by_user_id': row['created_by_user_id']
                    }
                
                return None
                
        except Exception as e:
            self.logger.error(f"Error getting workflow metadata: {e}")
            return None
    
    async def _delete_workflow_metadata(self, workflow_id: str, tenant_id: str) -> bool:
        """Delete workflow metadata from database"""
        try:
            if not self.pool_manager or not self.pool_manager.postgres_pool:
                return False
            
            async with self.pool_manager.postgres_pool.acquire() as conn:
                await conn.execute(f"SET app.current_tenant = '{tenant_id}'")
                
                query = """
                    DELETE FROM dsl_workflows
                    WHERE workflow_id = $1 AND tenant_id = $2
                """
                
                result = await conn.execute(query, workflow_id, tenant_id)
                
                # Check if any row was deleted
                return result.split()[-1] == '1'
                
        except Exception as e:
            self.logger.error(f"Error deleting workflow metadata: {e}")
            return False
    
    async def _upload_to_blob(self, workflow_id: str, yaml_content: str) -> str:
        """Upload workflow YAML to Azure Blob Storage"""
        try:
            if not self.blob_connection_string:
                # For development/testing without blob storage
                self.logger.warning("Blob storage not configured, using mock URL")
                return f"mock://workflows/{workflow_id}.yml"
            
            # TODO: Implement actual Azure Blob Storage upload
            # from azure.storage.blob import BlobServiceClient
            # blob_service_client = BlobServiceClient.from_connection_string(self.blob_connection_string)
            # blob_client = blob_service_client.get_blob_client(
            #     container=self.blob_container_name,
            #     blob=f"workflows/{workflow_id}.yml"
            # )
            # blob_client.upload_blob(yaml_content, overwrite=True)
            # return blob_client.url
            
            # Mock URL for now
            return f"https://mockblob.blob.core.windows.net/{self.blob_container_name}/workflows/{workflow_id}.yml"
            
        except Exception as e:
            self.logger.error(f"Error uploading to blob storage: {e}")
            raise
    
    async def _download_from_blob(self, blob_url: str) -> Optional[str]:
        """Download workflow YAML from Azure Blob Storage"""
        try:
            if blob_url.startswith("mock://"):
                # Return mock YAML content for testing
                return """
workflow_id: mock_workflow
name: Mock Workflow
module: Testing
automation_type: RBA
version: 1.0.0
steps:
  - id: mock_step
    type: notify
    params:
      channel: log
      message: "Mock workflow executed"
governance:
  policy_pack: default_policy
"""
            
            # TODO: Implement actual Azure Blob Storage download
            # from azure.storage.blob import BlobServiceClient
            # blob_service_client = BlobServiceClient.from_connection_string(self.blob_connection_string)
            # blob_client = blob_service_client.get_blob_client_from_url(blob_url)
            # return blob_client.download_blob().readall().decode('utf-8')
            
            # Mock content for now
            return None
            
        except Exception as e:
            self.logger.error(f"Error downloading from blob storage: {e}")
            return None
    
    async def _delete_from_blob(self, blob_url: str) -> bool:
        """Delete workflow YAML from Azure Blob Storage"""
        try:
            if blob_url.startswith("mock://"):
                self.logger.info(f"Mock blob deleted: {blob_url}")
                return True
            
            # TODO: Implement actual Azure Blob Storage deletion
            # from azure.storage.blob import BlobServiceClient
            # blob_service_client = BlobServiceClient.from_connection_string(self.blob_connection_string)
            # blob_client = blob_service_client.get_blob_client_from_url(blob_url)
            # blob_client.delete_blob()
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error deleting from blob storage: {e}")
            return False
