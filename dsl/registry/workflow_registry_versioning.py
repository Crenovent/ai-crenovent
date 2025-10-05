# Workflow Registry and Versioning System
# Tasks 7.3-T01 to T15: Workflow registry, versioning rules, metadata management

import json
import uuid
import hashlib
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Any, Optional, Union, Tuple
from dataclasses import dataclass, asdict, field
from enum import Enum
import logging
try:
    import semantic_version
except ImportError:
    # Mock semantic_version for environments where it's not available
    class MockVersion:
        def __init__(self, version_string):
            parts = version_string.split('.')
            self.major = int(parts[0]) if len(parts) > 0 else 0
            self.minor = int(parts[1]) if len(parts) > 1 else 0
            self.patch = int(parts[2]) if len(parts) > 2 else 0
        
        def __str__(self):
            return f"{self.major}.{self.minor}.{self.patch}"
        
        def __lt__(self, other):
            return (self.major, self.minor, self.patch) < (other.major, other.minor, other.patch)
        
        def __gt__(self, other):
            return (self.major, self.minor, self.patch) > (other.major, other.minor, other.patch)
        
        def __eq__(self, other):
            return (self.major, self.minor, self.patch) == (other.major, other.minor, other.patch)
        
        def next_major(self):
            return MockVersion(f"{self.major + 1}.0.0")
        
        def next_minor(self):
            return MockVersion(f"{self.major}.{self.minor + 1}.0")
        
        def next_patch(self):
            return MockVersion(f"{self.major}.{self.minor}.{self.patch + 1}")
    
    class semantic_version:
        Version = MockVersion

logger = logging.getLogger(__name__)

class WorkflowStatus(Enum):
    """Workflow lifecycle status"""
    DRAFT = "draft"
    UNDER_REVIEW = "under_review"
    APPROVED = "approved"
    PUBLISHED = "published"
    DEPRECATED = "deprecated"
    ARCHIVED = "archived"
    REJECTED = "rejected"

class VersionType(Enum):
    """Semantic version types"""
    MAJOR = "major"
    MINOR = "minor"
    PATCH = "patch"
    PRERELEASE = "prerelease"

class WorkflowCategory(Enum):
    """Workflow categories"""
    DATA_SYNC = "data_sync"
    PIPELINE_HYGIENE = "pipeline_hygiene"
    COMPENSATION = "compensation"
    REPORTING = "reporting"
    NOTIFICATION = "notification"
    APPROVAL = "approval"
    COMPLIANCE = "compliance"
    CUSTOM = "custom"

@dataclass
class WorkflowMetadata:
    """Comprehensive workflow metadata"""
    # All required fields first
    workflow_id: str
    name: str
    description: str
    category: WorkflowCategory
    version: str
    version_hash: str
    created_by: str
    created_at: str
    updated_by: str
    updated_at: str
    tenant_id: int
    industry_code: str
    tenant_tier: str
    status: WorkflowStatus
    
    # All optional fields with defaults
    previous_version: Optional[str] = None
    published_at: Optional[str] = None
    deprecated_at: Optional[str] = None
    
    # All optional fields with defaults
    compliance_frameworks: List[str] = field(default_factory=list)
    policy_pack_version: str = "default_v1.0"
    trust_score: float = 1.0
    approval_required: bool = False
    approved_by: Optional[str] = None
    approved_at: Optional[str] = None
    dsl_version: str = "1.0.0"
    schema_version: str = "1.0.0"
    runtime_requirements: Dict[str, Any] = field(default_factory=dict)
    dependencies: List[str] = field(default_factory=list)
    execution_count: int = 0
    success_rate: float = 1.0
    avg_execution_time_ms: int = 0
    last_executed_at: Optional[str] = None
    tags: List[str] = field(default_factory=list)
    keywords: List[str] = field(default_factory=list)
    documentation_url: Optional[str] = None
    changelog: List[Dict[str, Any]] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'WorkflowMetadata':
        """Create from dictionary"""
        # Handle enum conversions
        if 'category' in data:
            data['category'] = WorkflowCategory(data['category'])
        if 'status' in data:
            data['status'] = WorkflowStatus(data['status'])
        
        return cls(**data)

@dataclass
class WorkflowVersion:
    """Individual workflow version"""
    version: str
    version_hash: str
    dsl_content: Dict[str, Any]
    metadata: WorkflowMetadata
    created_at: str
    created_by: str
    change_summary: str
    breaking_changes: List[str] = field(default_factory=list)
    migration_notes: Optional[str] = None

@dataclass
class WorkflowRegistryEntry:
    """Complete workflow registry entry"""
    workflow_id: str
    current_version: str
    current_metadata: WorkflowMetadata
    versions: Dict[str, WorkflowVersion] = field(default_factory=dict)
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    updated_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

class SemanticVersionManager:
    """
    Semantic versioning manager for workflows
    Tasks 7.3-T03, T04: Version numbering and semantic versioning
    """
    
    def __init__(self):
        self.version_patterns = {
            VersionType.MAJOR: r'^\d+\.0\.0$',
            VersionType.MINOR: r'^\d+\.\d+\.0$',
            VersionType.PATCH: r'^\d+\.\d+\.\d+$',
            VersionType.PRERELEASE: r'^\d+\.\d+\.\d+-[a-zA-Z0-9\-\.]+$'
        }
    
    def parse_version(self, version_string: str) -> semantic_version.Version:
        """Parse version string into semantic version"""
        try:
            return semantic_version.Version(version_string)
        except ValueError as e:
            raise ValueError(f"Invalid version format: {version_string} - {e}")
    
    def increment_version(
        self,
        current_version: str,
        version_type: VersionType,
        prerelease_identifier: Optional[str] = None
    ) -> str:
        """Increment version based on type"""
        
        version = self.parse_version(current_version)
        
        if version_type == VersionType.MAJOR:
            new_version = version.next_major()
        elif version_type == VersionType.MINOR:
            new_version = version.next_minor()
        elif version_type == VersionType.PATCH:
            new_version = version.next_patch()
        elif version_type == VersionType.PRERELEASE:
            if prerelease_identifier:
                new_version = semantic_version.Version(
                    f"{version.major}.{version.minor}.{version.patch}-{prerelease_identifier}"
                )
            else:
                raise ValueError("Prerelease identifier required for prerelease versions")
        else:
            raise ValueError(f"Unknown version type: {version_type}")
        
        return str(new_version)
    
    def compare_versions(self, version1: str, version2: str) -> int:
        """Compare two versions (-1, 0, 1)"""
        v1 = self.parse_version(version1)
        v2 = self.parse_version(version2)
        
        if v1 < v2:
            return -1
        elif v1 > v2:
            return 1
        else:
            return 0
    
    def is_breaking_change(self, from_version: str, to_version: str) -> bool:
        """Check if version change is breaking"""
        v1 = self.parse_version(from_version)
        v2 = self.parse_version(to_version)
        
        # Major version increment is always breaking
        return v2.major > v1.major
    
    def get_compatible_versions(
        self,
        base_version: str,
        available_versions: List[str]
    ) -> List[str]:
        """Get versions compatible with base version"""
        
        base = self.parse_version(base_version)
        compatible = []
        
        for version_str in available_versions:
            try:
                version = self.parse_version(version_str)
                
                # Compatible if same major version and >= minor.patch
                if (version.major == base.major and 
                    (version.minor > base.minor or 
                     (version.minor == base.minor and version.patch >= base.patch))):
                    compatible.append(version_str)
                    
            except ValueError:
                continue  # Skip invalid versions
        
        return sorted(compatible, key=lambda v: self.parse_version(v), reverse=True)

class WorkflowHasher:
    """
    Workflow content hashing for integrity and change detection
    Tasks 7.3-T05, T06: Content hashing and change detection
    """
    
    def __init__(self):
        self.hash_algorithm = 'sha256'
    
    def calculate_workflow_hash(self, dsl_content: Dict[str, Any]) -> str:
        """Calculate deterministic hash of workflow DSL content"""
        
        # Normalize content for consistent hashing
        normalized_content = self._normalize_content(dsl_content)
        
        # Convert to JSON with sorted keys
        content_json = json.dumps(normalized_content, sort_keys=True, separators=(',', ':'))
        
        # Calculate hash
        hash_obj = hashlib.new(self.hash_algorithm)
        hash_obj.update(content_json.encode('utf-8'))
        
        return hash_obj.hexdigest()
    
    def calculate_metadata_hash(self, metadata: WorkflowMetadata) -> str:
        """Calculate hash of workflow metadata"""
        
        # Extract hashable metadata (exclude timestamps and counters)
        hashable_metadata = {
            'workflow_id': metadata.workflow_id,
            'name': metadata.name,
            'description': metadata.description,
            'category': metadata.category.value,
            'compliance_frameworks': sorted(metadata.compliance_frameworks),
            'policy_pack_version': metadata.policy_pack_version,
            'dependencies': sorted(metadata.dependencies),
            'tags': sorted(metadata.tags),
            'keywords': sorted(metadata.keywords)
        }
        
        content_json = json.dumps(hashable_metadata, sort_keys=True, separators=(',', ':'))
        
        hash_obj = hashlib.new(self.hash_algorithm)
        hash_obj.update(content_json.encode('utf-8'))
        
        return hash_obj.hexdigest()
    
    def _normalize_content(self, content: Dict[str, Any]) -> Dict[str, Any]:
        """Normalize content for consistent hashing"""
        
        if isinstance(content, dict):
            # Remove non-deterministic fields
            normalized = {}
            for key, value in content.items():
                if key not in ['created_at', 'updated_at', 'execution_id', 'trace_id']:
                    normalized[key] = self._normalize_content(value)
            return normalized
        
        elif isinstance(content, list):
            return [self._normalize_content(item) for item in content]
        
        else:
            return content
    
    def detect_changes(
        self,
        old_content: Dict[str, Any],
        new_content: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Detect changes between workflow versions"""
        
        old_hash = self.calculate_workflow_hash(old_content)
        new_hash = self.calculate_workflow_hash(new_content)
        
        if old_hash == new_hash:
            return {'has_changes': False, 'change_summary': 'No changes detected'}
        
        # Detailed change analysis
        changes = {
            'has_changes': True,
            'old_hash': old_hash,
            'new_hash': new_hash,
            'change_summary': 'Content modified',
            'field_changes': self._analyze_field_changes(old_content, new_content)
        }
        
        return changes
    
    def _analyze_field_changes(
        self,
        old_content: Dict[str, Any],
        new_content: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Analyze field-level changes"""
        
        changes = []
        
        # Find added fields
        old_keys = set(old_content.keys()) if isinstance(old_content, dict) else set()
        new_keys = set(new_content.keys()) if isinstance(new_content, dict) else set()
        
        added_keys = new_keys - old_keys
        removed_keys = old_keys - new_keys
        common_keys = old_keys & new_keys
        
        for key in added_keys:
            changes.append({
                'type': 'added',
                'field': key,
                'new_value': new_content[key]
            })
        
        for key in removed_keys:
            changes.append({
                'type': 'removed',
                'field': key,
                'old_value': old_content[key]
            })
        
        for key in common_keys:
            if old_content[key] != new_content[key]:
                changes.append({
                    'type': 'modified',
                    'field': key,
                    'old_value': old_content[key],
                    'new_value': new_content[key]
                })
        
        return changes

class WorkflowRegistryVersioning:
    """
    Comprehensive workflow registry and versioning system
    Tasks 7.3-T01 to T15: Registry, versioning, metadata management
    """
    
    def __init__(self, db_pool=None):
        self.db_pool = db_pool
        self.version_manager = SemanticVersionManager()
        self.hasher = WorkflowHasher()
        
        # In-memory registry (in practice would be database-backed)
        self.registry: Dict[str, WorkflowRegistryEntry] = {}
        
        # Workflow templates by industry
        self.industry_templates = {
            'SaaS': ['pipeline_hygiene', 'quota_rollover', 'churn_prevention'],
            'Banking': ['loan_sanction', 'kyc_validation', 'risk_assessment'],
            'Insurance': ['claims_processing', 'underwriting', 'fraud_detection'],
            'Healthcare': ['patient_onboarding', 'compliance_check', 'billing_validation']
        }
    
    async def register_workflow(
        self,
        workflow_id: str,
        name: str,
        description: str,
        category: WorkflowCategory,
        dsl_content: Dict[str, Any],
        tenant_id: int,
        industry_code: str,
        created_by: str,
        initial_version: str = "1.0.0"
    ) -> WorkflowRegistryEntry:
        """Register new workflow in registry"""
        
        # Calculate content hash
        version_hash = self.hasher.calculate_workflow_hash(dsl_content)
        
        # Create metadata
        metadata = WorkflowMetadata(
            workflow_id=workflow_id,
            name=name,
            description=description,
            category=category,
            version=initial_version,
            version_hash=version_hash,
            created_by=created_by,
            created_at=datetime.now(timezone.utc).isoformat(),
            updated_by=created_by,
            updated_at=datetime.now(timezone.utc).isoformat(),
            status=WorkflowStatus.DRAFT,
            tenant_id=tenant_id,
            industry_code=industry_code,
            tenant_tier="professional"  # Default
        )
        
        # Create initial version
        initial_workflow_version = WorkflowVersion(
            version=initial_version,
            version_hash=version_hash,
            dsl_content=dsl_content,
            metadata=metadata,
            created_at=datetime.now(timezone.utc).isoformat(),
            created_by=created_by,
            change_summary="Initial version"
        )
        
        # Create registry entry
        registry_entry = WorkflowRegistryEntry(
            workflow_id=workflow_id,
            current_version=initial_version,
            current_metadata=metadata,
            versions={initial_version: initial_workflow_version}
        )
        
        # Store in registry
        self.registry[workflow_id] = registry_entry
        
        # Persist to database if available
        if self.db_pool:
            await self._persist_registry_entry(registry_entry)
        
        logger.info(f"Registered workflow: {workflow_id} v{initial_version}")
        return registry_entry
    
    async def create_new_version(
        self,
        workflow_id: str,
        version_type: VersionType,
        dsl_content: Dict[str, Any],
        updated_by: str,
        change_summary: str,
        breaking_changes: List[str] = None,
        prerelease_identifier: Optional[str] = None
    ) -> WorkflowVersion:
        """Create new version of existing workflow"""
        
        if workflow_id not in self.registry:
            raise ValueError(f"Workflow not found: {workflow_id}")
        
        registry_entry = self.registry[workflow_id]
        current_version = registry_entry.current_version
        
        # Generate new version number
        new_version = self.version_manager.increment_version(
            current_version, version_type, prerelease_identifier
        )
        
        # Calculate new content hash
        new_version_hash = self.hasher.calculate_workflow_hash(dsl_content)
        
        # Check if content actually changed
        current_workflow_version = registry_entry.versions[current_version]
        if new_version_hash == current_workflow_version.version_hash:
            raise ValueError("No content changes detected - version not created")
        
        # Detect changes
        change_analysis = self.hasher.detect_changes(
            current_workflow_version.dsl_content,
            dsl_content
        )
        
        # Create new metadata
        new_metadata = WorkflowMetadata(
            **asdict(registry_entry.current_metadata),
            version=new_version,
            version_hash=new_version_hash,
            previous_version=current_version,
            updated_by=updated_by,
            updated_at=datetime.now(timezone.utc).isoformat(),
            status=WorkflowStatus.DRAFT,
            execution_count=0,  # Reset for new version
            success_rate=1.0,
            avg_execution_time_ms=0,
            last_executed_at=None
        )
        
        # Add changelog entry
        changelog_entry = {
            'version': new_version,
            'date': datetime.now(timezone.utc).isoformat(),
            'author': updated_by,
            'summary': change_summary,
            'breaking_changes': breaking_changes or [],
            'change_analysis': change_analysis
        }
        new_metadata.changelog.append(changelog_entry)
        
        # Create new version
        new_workflow_version = WorkflowVersion(
            version=new_version,
            version_hash=new_version_hash,
            dsl_content=dsl_content,
            metadata=new_metadata,
            created_at=datetime.now(timezone.utc).isoformat(),
            created_by=updated_by,
            change_summary=change_summary,
            breaking_changes=breaking_changes or []
        )
        
        # Update registry entry
        registry_entry.versions[new_version] = new_workflow_version
        registry_entry.current_version = new_version
        registry_entry.current_metadata = new_metadata
        registry_entry.updated_at = datetime.now(timezone.utc).isoformat()
        
        # Persist to database if available
        if self.db_pool:
            await self._persist_workflow_version(workflow_id, new_workflow_version)
        
        logger.info(f"Created new version: {workflow_id} v{new_version}")
        return new_workflow_version
    
    async def publish_workflow(
        self,
        workflow_id: str,
        version: Optional[str] = None,
        published_by: str = "system"
    ) -> bool:
        """Publish workflow version"""
        
        if workflow_id not in self.registry:
            raise ValueError(f"Workflow not found: {workflow_id}")
        
        registry_entry = self.registry[workflow_id]
        target_version = version or registry_entry.current_version
        
        if target_version not in registry_entry.versions:
            raise ValueError(f"Version not found: {target_version}")
        
        workflow_version = registry_entry.versions[target_version]
        
        # Update status to published
        workflow_version.metadata.status = WorkflowStatus.PUBLISHED
        workflow_version.metadata.published_at = datetime.now(timezone.utc).isoformat()
        workflow_version.metadata.updated_by = published_by
        workflow_version.metadata.updated_at = datetime.now(timezone.utc).isoformat()
        
        # Update current metadata if this is the current version
        if target_version == registry_entry.current_version:
            registry_entry.current_metadata = workflow_version.metadata
        
        # Persist to database if available
        if self.db_pool:
            await self._update_workflow_status(workflow_id, target_version, WorkflowStatus.PUBLISHED)
        
        logger.info(f"Published workflow: {workflow_id} v{target_version}")
        return True
    
    async def deprecate_workflow(
        self,
        workflow_id: str,
        version: Optional[str] = None,
        deprecated_by: str = "system",
        reason: str = "Deprecated"
    ) -> bool:
        """Deprecate workflow version"""
        
        if workflow_id not in self.registry:
            raise ValueError(f"Workflow not found: {workflow_id}")
        
        registry_entry = self.registry[workflow_id]
        target_version = version or registry_entry.current_version
        
        if target_version not in registry_entry.versions:
            raise ValueError(f"Version not found: {target_version}")
        
        workflow_version = registry_entry.versions[target_version]
        
        # Update status to deprecated
        workflow_version.metadata.status = WorkflowStatus.DEPRECATED
        workflow_version.metadata.deprecated_at = datetime.now(timezone.utc).isoformat()
        workflow_version.metadata.updated_by = deprecated_by
        workflow_version.metadata.updated_at = datetime.now(timezone.utc).isoformat()
        
        # Add to changelog
        changelog_entry = {
            'version': target_version,
            'date': datetime.now(timezone.utc).isoformat(),
            'author': deprecated_by,
            'summary': f"Deprecated: {reason}",
            'breaking_changes': [],
            'change_analysis': {'has_changes': False}
        }
        workflow_version.metadata.changelog.append(changelog_entry)
        
        # Update current metadata if this is the current version
        if target_version == registry_entry.current_version:
            registry_entry.current_metadata = workflow_version.metadata
        
        # Persist to database if available
        if self.db_pool:
            await self._update_workflow_status(workflow_id, target_version, WorkflowStatus.DEPRECATED)
        
        logger.info(f"Deprecated workflow: {workflow_id} v{target_version}")
        return True
    
    async def get_workflow(
        self,
        workflow_id: str,
        version: Optional[str] = None
    ) -> Optional[WorkflowVersion]:
        """Get workflow by ID and version"""
        
        if workflow_id not in self.registry:
            return None
        
        registry_entry = self.registry[workflow_id]
        target_version = version or registry_entry.current_version
        
        return registry_entry.versions.get(target_version)
    
    async def list_workflows(
        self,
        tenant_id: Optional[int] = None,
        industry_code: Optional[str] = None,
        category: Optional[WorkflowCategory] = None,
        status: Optional[WorkflowStatus] = None
    ) -> List[WorkflowMetadata]:
        """List workflows with optional filters"""
        
        workflows = []
        
        for registry_entry in self.registry.values():
            metadata = registry_entry.current_metadata
            
            # Apply filters
            if tenant_id and metadata.tenant_id != tenant_id:
                continue
            
            if industry_code and metadata.industry_code != industry_code:
                continue
            
            if category and metadata.category != category:
                continue
            
            if status and metadata.status != status:
                continue
            
            workflows.append(metadata)
        
        # Sort by updated_at descending
        workflows.sort(key=lambda w: w.updated_at, reverse=True)
        
        return workflows
    
    async def get_workflow_versions(self, workflow_id: str) -> List[WorkflowVersion]:
        """Get all versions of a workflow"""
        
        if workflow_id not in self.registry:
            return []
        
        registry_entry = self.registry[workflow_id]
        versions = list(registry_entry.versions.values())
        
        # Sort by version (newest first)
        versions.sort(
            key=lambda v: self.version_manager.parse_version(v.version),
            reverse=True
        )
        
        return versions
    
    async def update_execution_metrics(
        self,
        workflow_id: str,
        version: str,
        execution_time_ms: int,
        success: bool
    ) -> None:
        """Update workflow execution metrics"""
        
        if workflow_id not in self.registry:
            return
        
        registry_entry = self.registry[workflow_id]
        
        if version not in registry_entry.versions:
            return
        
        workflow_version = registry_entry.versions[version]
        metadata = workflow_version.metadata
        
        # Update metrics
        metadata.execution_count += 1
        metadata.last_executed_at = datetime.now(timezone.utc).isoformat()
        
        # Update success rate (exponential moving average)
        if metadata.execution_count == 1:
            metadata.success_rate = 1.0 if success else 0.0
        else:
            alpha = 0.1  # Smoothing factor
            new_success = 1.0 if success else 0.0
            metadata.success_rate = alpha * new_success + (1 - alpha) * metadata.success_rate
        
        # Update average execution time (exponential moving average)
        if metadata.execution_count == 1:
            metadata.avg_execution_time_ms = execution_time_ms
        else:
            alpha = 0.1
            metadata.avg_execution_time_ms = int(
                alpha * execution_time_ms + (1 - alpha) * metadata.avg_execution_time_ms
            )
        
        # Update current metadata if this is the current version
        if version == registry_entry.current_version:
            registry_entry.current_metadata = metadata
        
        # Persist to database if available
        if self.db_pool:
            await self._update_execution_metrics(workflow_id, version, metadata)
    
    async def search_workflows(
        self,
        query: str,
        tenant_id: Optional[int] = None,
        industry_code: Optional[str] = None
    ) -> List[WorkflowMetadata]:
        """Search workflows by name, description, or tags"""
        
        query_lower = query.lower()
        matching_workflows = []
        
        for registry_entry in self.registry.values():
            metadata = registry_entry.current_metadata
            
            # Apply tenant/industry filters
            if tenant_id and metadata.tenant_id != tenant_id:
                continue
            
            if industry_code and metadata.industry_code != industry_code:
                continue
            
            # Search in name, description, tags, keywords
            searchable_text = ' '.join([
                metadata.name.lower(),
                metadata.description.lower(),
                ' '.join(metadata.tags).lower(),
                ' '.join(metadata.keywords).lower()
            ])
            
            if query_lower in searchable_text:
                matching_workflows.append(metadata)
        
        # Sort by relevance (simple scoring)
        def relevance_score(metadata: WorkflowMetadata) -> float:
            score = 0.0
            
            if query_lower in metadata.name.lower():
                score += 10.0
            
            if query_lower in metadata.description.lower():
                score += 5.0
            
            if query_lower in ' '.join(metadata.tags).lower():
                score += 3.0
            
            if query_lower in ' '.join(metadata.keywords).lower():
                score += 1.0
            
            # Boost by trust score and execution count
            score += metadata.trust_score * 2.0
            score += min(metadata.execution_count / 100.0, 5.0)
            
            return score
        
        matching_workflows.sort(key=relevance_score, reverse=True)
        
        return matching_workflows
    
    async def get_registry_statistics(self) -> Dict[str, Any]:
        """Get registry statistics"""
        
        stats = {
            'total_workflows': len(self.registry),
            'total_versions': 0,
            'workflows_by_status': {},
            'workflows_by_category': {},
            'workflows_by_industry': {},
            'average_versions_per_workflow': 0.0,
            'most_executed_workflows': [],
            'highest_trust_score_workflows': []
        }
        
        all_workflows = []
        
        for registry_entry in self.registry.values():
            stats['total_versions'] += len(registry_entry.versions)
            metadata = registry_entry.current_metadata
            all_workflows.append(metadata)
            
            # Count by status
            status = metadata.status.value
            stats['workflows_by_status'][status] = stats['workflows_by_status'].get(status, 0) + 1
            
            # Count by category
            category = metadata.category.value
            stats['workflows_by_category'][category] = stats['workflows_by_category'].get(category, 0) + 1
            
            # Count by industry
            industry = metadata.industry_code
            stats['workflows_by_industry'][industry] = stats['workflows_by_industry'].get(industry, 0) + 1
        
        # Calculate averages
        if len(self.registry) > 0:
            stats['average_versions_per_workflow'] = stats['total_versions'] / len(self.registry)
        
        # Top workflows by execution count
        stats['most_executed_workflows'] = sorted(
            all_workflows,
            key=lambda w: w.execution_count,
            reverse=True
        )[:10]
        
        # Top workflows by trust score
        stats['highest_trust_score_workflows'] = sorted(
            all_workflows,
            key=lambda w: w.trust_score,
            reverse=True
        )[:10]
        
        return stats
    
    async def _persist_registry_entry(self, registry_entry: WorkflowRegistryEntry) -> None:
        """Persist registry entry to database"""
        # Mock implementation - would use actual database
        pass
    
    async def _persist_workflow_version(self, workflow_id: str, version: WorkflowVersion) -> None:
        """Persist workflow version to database"""
        # Mock implementation - would use actual database
        pass
    
    async def _update_workflow_status(
        self,
        workflow_id: str,
        version: str,
        status: WorkflowStatus
    ) -> None:
        """Update workflow status in database"""
        # Mock implementation - would use actual database
        pass
    
    async def _update_execution_metrics(
        self,
        workflow_id: str,
        version: str,
        metadata: WorkflowMetadata
    ) -> None:
        """Update execution metrics in database"""
        # Mock implementation - would use actual database
        pass

# Global workflow registry
workflow_registry = WorkflowRegistryVersioning()
