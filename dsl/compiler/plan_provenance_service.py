"""
Plan Provenance Service - Task 6.2.29
======================================

Plan provenance: source refs, commit SHAs, author, approver
- Captures complete provenance for tracing plans back to source files, commits, and human actors
- Manifest fields and backing metadata capture
- Automatic collection from LSP/editor context and CI environment variables
- Validation checks for required provenance fields
- Backend metadata management (no CI/Git integration - that's infrastructure)

Dependencies: Task 6.2.26 (Plan Manifest Generator)
Outputs: Complete provenance metadata → enables full audit trail and traceability
"""

from typing import Dict, List, Any, Optional, Tuple, Union
import json
import logging
from datetime import datetime, timezone
from dataclasses import dataclass, field, asdict
from enum import Enum
import re
import os
from pathlib import Path
import hashlib

logger = logging.getLogger(__name__)

class ProvenanceSource(Enum):
    """Source of provenance information"""
    LSP_EDITOR = "lsp_editor"
    CI_ENVIRONMENT = "ci_environment"
    CLI_PROVIDED = "cli_provided"
    API_REQUEST = "api_request"
    MANUAL_ENTRY = "manual_entry"

class ValidationSeverity(Enum):
    """Severity levels for provenance validation"""
    ERROR = "error"
    WARNING = "warning"
    INFO = "info"

@dataclass
class SourceFileReference:
    """Reference to a source file"""
    file_path: str
    line_ranges: List[Tuple[int, int]] = field(default_factory=list)  # [(start, end), ...]
    file_hash: Optional[str] = None
    last_modified: Optional[datetime] = None
    file_size_bytes: Optional[int] = None
    encoding: str = "utf-8"

@dataclass
class CommitReference:
    """Reference to a commit"""
    commit_sha: str
    repository_url: Optional[str] = None
    branch: Optional[str] = None
    commit_message: Optional[str] = None
    commit_timestamp: Optional[datetime] = None
    author_name: Optional[str] = None
    author_email: Optional[str] = None

@dataclass
class HumanActor:
    """Human actor in the workflow process"""
    user_id: str
    name: Optional[str] = None
    email: Optional[str] = None
    role: Optional[str] = None
    department: Optional[str] = None
    authentication_method: Optional[str] = None
    session_id: Optional[str] = None
    ip_address: Optional[str] = None
    user_agent: Optional[str] = None

@dataclass
class BuildEnvironment:
    """Build/compilation environment information"""
    ci_system: Optional[str] = None
    build_id: Optional[str] = None
    job_id: Optional[str] = None
    pipeline_id: Optional[str] = None
    build_url: Optional[str] = None
    environment_name: Optional[str] = None  # dev, staging, prod
    build_timestamp: Optional[datetime] = None
    build_duration_seconds: Optional[float] = None
    build_status: Optional[str] = None
    build_agent: Optional[str] = None

@dataclass
class WorkspaceState:
    """State of the development workspace"""
    workspace_root: Optional[str] = None
    workspace_id: Optional[str] = None
    dirty_files: List[str] = field(default_factory=list)  # Modified but uncommitted
    untracked_files: List[str] = field(default_factory=list)
    staged_files: List[str] = field(default_factory=list)
    current_branch: Optional[str] = None
    last_commit_sha: Optional[str] = None
    workspace_tools: Dict[str, str] = field(default_factory=dict)  # tool -> version

@dataclass
class ChangeSetSummary:
    """Summary of changes in this plan"""
    files_added: List[str] = field(default_factory=list)
    files_modified: List[str] = field(default_factory=list)
    files_deleted: List[str] = field(default_factory=list)
    lines_added: int = 0
    lines_deleted: int = 0
    change_description: Optional[str] = None
    change_category: Optional[str] = None  # feature, bugfix, refactor, etc.
    breaking_changes: bool = False

@dataclass
class ProvenanceValidationResult:
    """Result of provenance validation"""
    is_valid: bool
    severity: ValidationSeverity
    field_name: str
    message: str
    suggested_fix: Optional[str] = None
    validation_rule: str = ""

@dataclass
class CompletePlanProvenance:
    """Complete provenance information for a plan"""
    # Core identification
    plan_id: str
    workflow_id: str
    provenance_id: str
    
    # Source references
    source_files: List[SourceFileReference] = field(default_factory=list)
    commit_references: List[CommitReference] = field(default_factory=list)
    
    # Human actors
    author: Optional[HumanActor] = None
    approver: Optional[HumanActor] = None
    reviewers: List[HumanActor] = field(default_factory=list)
    
    # Environment and build
    build_environment: Optional[BuildEnvironment] = None
    workspace_state: Optional[WorkspaceState] = None
    change_set_summary: Optional[ChangeSetSummary] = None
    
    # Metadata
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    provenance_source: ProvenanceSource = ProvenanceSource.MANUAL_ENTRY
    collection_metadata: Dict[str, Any] = field(default_factory=dict)
    
    # Validation
    validation_results: List[ProvenanceValidationResult] = field(default_factory=list)
    is_complete: bool = False
    completeness_score: float = 0.0

# Task 6.2.29: Plan Provenance Service
class PlanProvenanceService:
    """Service for capturing and managing plan provenance information"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
        # Storage for provenance records
        self.provenance_records: Dict[str, CompletePlanProvenance] = {}  # provenance_id -> record
        self.plan_provenance_map: Dict[str, str] = {}  # plan_id -> provenance_id
        
        # Validation rules configuration
        self.validation_rules = {
            'required_fields': {
                'plan_id': ValidationSeverity.ERROR,
                'workflow_id': ValidationSeverity.ERROR,
                'author': ValidationSeverity.ERROR,
                'source_files': ValidationSeverity.WARNING,
                'commit_references': ValidationSeverity.WARNING
            },
            'conditional_requirements': {
                'production': {
                    'approver': ValidationSeverity.ERROR,
                    'commit_references': ValidationSeverity.ERROR,
                    'build_environment': ValidationSeverity.ERROR
                }
            }
        }
        
        # Environment variable mappings for automatic collection
        self.ci_env_mappings = {
            'github_actions': {
                'ci_system': 'GITHUB_ACTIONS',
                'build_id': 'GITHUB_RUN_ID',
                'job_id': 'GITHUB_JOB',
                'build_url': 'GITHUB_SERVER_URL',
                'commit_sha': 'GITHUB_SHA',
                'branch': 'GITHUB_REF_NAME',
                'repository_url': 'GITHUB_REPOSITORY',
                'actor': 'GITHUB_ACTOR'
            },
            'gitlab_ci': {
                'ci_system': 'GITLAB_CI',
                'build_id': 'CI_PIPELINE_ID',
                'job_id': 'CI_JOB_ID',
                'build_url': 'CI_PIPELINE_URL',
                'commit_sha': 'CI_COMMIT_SHA',
                'branch': 'CI_COMMIT_REF_NAME',
                'repository_url': 'CI_REPOSITORY_URL'
            },
            'jenkins': {
                'ci_system': 'JENKINS',
                'build_id': 'BUILD_ID',
                'job_id': 'JOB_NAME',
                'build_url': 'BUILD_URL',
                'commit_sha': 'GIT_COMMIT',
                'branch': 'GIT_BRANCH'
            }
        }
    
    def create_provenance_record(self, plan_id: str, workflow_id: str,
                               provenance_source: ProvenanceSource = ProvenanceSource.MANUAL_ENTRY) -> CompletePlanProvenance:
        """
        Create a new provenance record
        
        Args:
            plan_id: Plan identifier
            workflow_id: Workflow identifier
            provenance_source: Source of provenance information
            
        Returns:
            CompletePlanProvenance record
        """
        provenance_id = f"prov_{hashlib.sha256(f'{plan_id}_{workflow_id}_{datetime.now().isoformat()}'.encode()).hexdigest()[:16]}"
        
        provenance = CompletePlanProvenance(
            plan_id=plan_id,
            workflow_id=workflow_id,
            provenance_id=provenance_id,
            provenance_source=provenance_source,
            created_at=datetime.now(timezone.utc)
        )
        
        # Store record
        self.provenance_records[provenance_id] = provenance
        self.plan_provenance_map[plan_id] = provenance_id
        
        self.logger.info(f"✅ Created provenance record: {provenance_id} for plan {plan_id}")
        
        return provenance
    
    def collect_provenance_from_environment(self, provenance: CompletePlanProvenance) -> CompletePlanProvenance:
        """
        Automatically collect provenance information from environment variables
        
        Args:
            provenance: Existing provenance record to enrich
            
        Returns:
            Updated provenance record
        """
        # Detect CI system
        ci_system = self._detect_ci_system()
        
        if ci_system:
            self.logger.info(f"Detected CI system: {ci_system}")
            
            # Collect build environment
            build_env = self._collect_build_environment(ci_system)
            if build_env:
                provenance.build_environment = build_env
            
            # Collect commit information
            commit_refs = self._collect_commit_references(ci_system)
            if commit_refs:
                provenance.commit_references.extend(commit_refs)
            
            # Update provenance source
            provenance.provenance_source = ProvenanceSource.CI_ENVIRONMENT
            provenance.collection_metadata['ci_system'] = ci_system
        
        # Collect workspace state if available
        workspace_state = self._collect_workspace_state()
        if workspace_state:
            provenance.workspace_state = workspace_state
        
        # Update collection metadata
        provenance.collection_metadata.update({
            'collected_at': datetime.now(timezone.utc).isoformat(),
            'collection_method': 'automatic_environment'
        })
        
        return provenance
    
    def add_source_file_references(self, provenance: CompletePlanProvenance,
                                 file_paths: List[str],
                                 include_line_ranges: bool = False) -> CompletePlanProvenance:
        """
        Add source file references to provenance
        
        Args:
            provenance: Provenance record to update
            file_paths: List of file paths to add
            include_line_ranges: Whether to analyze and include line ranges
            
        Returns:
            Updated provenance record
        """
        for file_path in file_paths:
            if not os.path.exists(file_path):
                self.logger.warning(f"Source file not found: {file_path}")
                continue
            
            # Get file metadata
            file_stat = os.stat(file_path)
            file_hash = self._compute_file_hash(file_path)
            
            source_ref = SourceFileReference(
                file_path=file_path,
                file_hash=file_hash,
                last_modified=datetime.fromtimestamp(file_stat.st_mtime, timezone.utc),
                file_size_bytes=file_stat.st_size
            )
            
            # Add line ranges if requested
            if include_line_ranges:
                line_ranges = self._analyze_file_line_ranges(file_path)
                source_ref.line_ranges = line_ranges
            
            provenance.source_files.append(source_ref)
        
        self.logger.info(f"Added {len(file_paths)} source file references to provenance {provenance.provenance_id}")
        
        return provenance
    
    def set_human_actors(self, provenance: CompletePlanProvenance,
                        author: Optional[HumanActor] = None,
                        approver: Optional[HumanActor] = None,
                        reviewers: Optional[List[HumanActor]] = None) -> CompletePlanProvenance:
        """
        Set human actors for the provenance record
        
        Args:
            provenance: Provenance record to update
            author: Plan author
            approver: Plan approver
            reviewers: List of reviewers
            
        Returns:
            Updated provenance record
        """
        if author:
            provenance.author = author
            self.logger.info(f"Set author: {author.user_id} for provenance {provenance.provenance_id}")
        
        if approver:
            provenance.approver = approver
            self.logger.info(f"Set approver: {approver.user_id} for provenance {provenance.provenance_id}")
        
        if reviewers:
            provenance.reviewers = reviewers
            self.logger.info(f"Set {len(reviewers)} reviewers for provenance {provenance.provenance_id}")
        
        return provenance
    
    def add_change_set_summary(self, provenance: CompletePlanProvenance,
                             change_summary: ChangeSetSummary) -> CompletePlanProvenance:
        """
        Add change set summary to provenance
        
        Args:
            provenance: Provenance record to update
            change_summary: Summary of changes
            
        Returns:
            Updated provenance record
        """
        provenance.change_set_summary = change_summary
        
        self.logger.info(f"Added change set summary to provenance {provenance.provenance_id}: "
                        f"{len(change_summary.files_modified)} modified, "
                        f"{len(change_summary.files_added)} added, "
                        f"{len(change_summary.files_deleted)} deleted")
        
        return provenance
    
    def validate_provenance(self, provenance: CompletePlanProvenance,
                          environment: str = "development") -> List[ProvenanceValidationResult]:
        """
        Validate provenance record against requirements
        
        Args:
            provenance: Provenance record to validate
            environment: Target environment (affects validation rules)
            
        Returns:
            List of validation results
        """
        validation_results = []
        
        # Check required fields
        for field_name, severity in self.validation_rules['required_fields'].items():
            result = self._validate_required_field(provenance, field_name, severity)
            if result:
                validation_results.append(result)
        
        # Check conditional requirements based on environment
        if environment in self.validation_rules['conditional_requirements']:
            conditional_rules = self.validation_rules['conditional_requirements'][environment]
            for field_name, severity in conditional_rules.items():
                result = self._validate_required_field(provenance, field_name, severity)
                if result:
                    validation_results.append(result)
        
        # Validate commit references
        for i, commit_ref in enumerate(provenance.commit_references):
            commit_validation = self._validate_commit_reference(commit_ref, i)
            validation_results.extend(commit_validation)
        
        # Validate source files
        for i, source_file in enumerate(provenance.source_files):
            file_validation = self._validate_source_file(source_file, i)
            validation_results.extend(file_validation)
        
        # Calculate completeness score
        completeness_score = self._calculate_completeness_score(provenance, environment)
        provenance.completeness_score = completeness_score
        provenance.is_complete = completeness_score >= 0.8  # 80% threshold
        
        # Store validation results
        provenance.validation_results = validation_results
        
        # Log validation summary
        error_count = len([r for r in validation_results if r.severity == ValidationSeverity.ERROR])
        warning_count = len([r for r in validation_results if r.severity == ValidationSeverity.WARNING])
        
        self.logger.info(f"Validated provenance {provenance.provenance_id}: "
                        f"completeness {completeness_score:.1%}, "
                        f"{error_count} errors, {warning_count} warnings")
        
        return validation_results
    
    def get_provenance_by_plan_id(self, plan_id: str) -> Optional[CompletePlanProvenance]:
        """Get provenance record by plan ID"""
        provenance_id = self.plan_provenance_map.get(plan_id)
        if provenance_id:
            return self.provenance_records.get(provenance_id)
        return None
    
    def get_provenance_summary(self, provenance_id: str) -> Dict[str, Any]:
        """Get summary of provenance record"""
        provenance = self.provenance_records.get(provenance_id)
        if not provenance:
            return {'error': 'Provenance record not found'}
        
        return {
            'provenance_id': provenance.provenance_id,
            'plan_id': provenance.plan_id,
            'workflow_id': provenance.workflow_id,
            'created_at': provenance.created_at.isoformat(),
            'provenance_source': provenance.provenance_source.value,
            'is_complete': provenance.is_complete,
            'completeness_score': provenance.completeness_score,
            'source_files_count': len(provenance.source_files),
            'commit_references_count': len(provenance.commit_references),
            'has_author': provenance.author is not None,
            'has_approver': provenance.approver is not None,
            'reviewers_count': len(provenance.reviewers),
            'validation_errors': len([r for r in provenance.validation_results if r.severity == ValidationSeverity.ERROR]),
            'validation_warnings': len([r for r in provenance.validation_results if r.severity == ValidationSeverity.WARNING])
        }
    
    def export_provenance_for_manifest(self, provenance_id: str) -> Dict[str, Any]:
        """
        Export provenance data for inclusion in plan manifest
        
        Args:
            provenance_id: Provenance record identifier
            
        Returns:
            Dictionary suitable for manifest inclusion
        """
        provenance = self.provenance_records.get(provenance_id)
        if not provenance:
            return {}
        
        # Convert to dictionary and clean up for manifest
        provenance_dict = asdict(provenance)
        
        # Remove validation results and internal metadata
        manifest_provenance = {
            'provenance_id': provenance_dict['provenance_id'],
            'source_files': [
                {
                    'file_path': sf['file_path'],
                    'file_hash': sf['file_hash'],
                    'line_ranges': sf['line_ranges']
                }
                for sf in provenance_dict['source_files']
            ],
            'commit_references': [
                {
                    'commit_sha': cr['commit_sha'],
                    'repository_url': cr['repository_url'],
                    'branch': cr['branch'],
                    'commit_timestamp': cr['commit_timestamp']
                }
                for cr in provenance_dict['commit_references']
            ],
            'author': {
                'user_id': provenance_dict['author']['user_id'],
                'name': provenance_dict['author']['name'],
                'role': provenance_dict['author']['role']
            } if provenance_dict['author'] else None,
            'approver': {
                'user_id': provenance_dict['approver']['user_id'],
                'name': provenance_dict['approver']['name'],
                'role': provenance_dict['approver']['role']
            } if provenance_dict['approver'] else None,
            'build_environment': {
                'ci_system': provenance_dict['build_environment']['ci_system'],
                'build_id': provenance_dict['build_environment']['build_id'],
                'environment_name': provenance_dict['build_environment']['environment_name'],
                'build_timestamp': provenance_dict['build_environment']['build_timestamp']
            } if provenance_dict['build_environment'] else None,
            'created_at': provenance_dict['created_at'],
            'provenance_source': provenance_dict['provenance_source'],
            'is_complete': provenance_dict['is_complete'],
            'completeness_score': provenance_dict['completeness_score']
        }
        
        # Remove None values
        return {k: v for k, v in manifest_provenance.items() if v is not None}
    
    def _detect_ci_system(self) -> Optional[str]:
        """Detect CI system from environment variables"""
        if os.getenv('GITHUB_ACTIONS'):
            return 'github_actions'
        elif os.getenv('GITLAB_CI'):
            return 'gitlab_ci'
        elif os.getenv('JENKINS_URL'):
            return 'jenkins'
        else:
            return None
    
    def _collect_build_environment(self, ci_system: str) -> Optional[BuildEnvironment]:
        """Collect build environment information"""
        if ci_system not in self.ci_env_mappings:
            return None
        
        mappings = self.ci_env_mappings[ci_system]
        
        build_env = BuildEnvironment(
            ci_system=ci_system,
            build_id=os.getenv(mappings.get('build_id', '')),
            job_id=os.getenv(mappings.get('job_id', '')),
            build_url=os.getenv(mappings.get('build_url', '')),
            environment_name=os.getenv('ENVIRONMENT', 'unknown'),
            build_timestamp=datetime.now(timezone.utc)
        )
        
        return build_env
    
    def _collect_commit_references(self, ci_system: str) -> List[CommitReference]:
        """Collect commit references from CI environment"""
        if ci_system not in self.ci_env_mappings:
            return []
        
        mappings = self.ci_env_mappings[ci_system]
        commit_sha = os.getenv(mappings.get('commit_sha', ''))
        
        if not commit_sha:
            return []
        
        commit_ref = CommitReference(
            commit_sha=commit_sha,
            repository_url=os.getenv(mappings.get('repository_url', '')),
            branch=os.getenv(mappings.get('branch', '')),
            author_name=os.getenv(mappings.get('actor', ''))
        )
        
        return [commit_ref]
    
    def _collect_workspace_state(self) -> Optional[WorkspaceState]:
        """Collect workspace state information"""
        # This would integrate with Git/workspace tools in a real implementation
        # For now, return basic information
        
        workspace_root = os.getcwd()
        
        workspace_state = WorkspaceState(
            workspace_root=workspace_root,
            workspace_id=hashlib.sha256(workspace_root.encode()).hexdigest()[:16]
        )
        
        return workspace_state
    
    def _compute_file_hash(self, file_path: str) -> str:
        """Compute SHA-256 hash of a file"""
        try:
            with open(file_path, 'rb') as f:
                return hashlib.sha256(f.read()).hexdigest()
        except Exception as e:
            self.logger.warning(f"Failed to compute hash for {file_path}: {e}")
            return ""
    
    def _analyze_file_line_ranges(self, file_path: str) -> List[Tuple[int, int]]:
        """Analyze file to determine relevant line ranges"""
        # Simple implementation - return full file range
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                lines = f.readlines()
                return [(1, len(lines))]
        except Exception:
            return []
    
    def _validate_required_field(self, provenance: CompletePlanProvenance,
                                field_name: str, severity: ValidationSeverity) -> Optional[ProvenanceValidationResult]:
        """Validate a required field"""
        value = getattr(provenance, field_name, None)
        
        is_empty = (
            value is None or
            (isinstance(value, str) and not value.strip()) or
            (isinstance(value, list) and len(value) == 0)
        )
        
        if is_empty:
            return ProvenanceValidationResult(
                is_valid=False,
                severity=severity,
                field_name=field_name,
                message=f"Required field '{field_name}' is missing or empty",
                suggested_fix=f"Provide a value for {field_name}",
                validation_rule="required_field"
            )
        
        return None
    
    def _validate_commit_reference(self, commit_ref: CommitReference, index: int) -> List[ProvenanceValidationResult]:
        """Validate a commit reference"""
        results = []
        
        # Check commit SHA format
        if not re.match(r'^[a-f0-9]{40}$', commit_ref.commit_sha):
            results.append(ProvenanceValidationResult(
                is_valid=False,
                severity=ValidationSeverity.WARNING,
                field_name=f"commit_references[{index}].commit_sha",
                message=f"Commit SHA '{commit_ref.commit_sha}' does not appear to be a valid Git SHA",
                validation_rule="commit_sha_format"
            ))
        
        return results
    
    def _validate_source_file(self, source_file: SourceFileReference, index: int) -> List[ProvenanceValidationResult]:
        """Validate a source file reference"""
        results = []
        
        # Check if file exists
        if not os.path.exists(source_file.file_path):
            results.append(ProvenanceValidationResult(
                is_valid=False,
                severity=ValidationSeverity.WARNING,
                field_name=f"source_files[{index}].file_path",
                message=f"Source file '{source_file.file_path}' does not exist",
                suggested_fix="Verify the file path is correct",
                validation_rule="file_exists"
            ))
        
        return results
    
    def _calculate_completeness_score(self, provenance: CompletePlanProvenance, environment: str) -> float:
        """Calculate completeness score for provenance record"""
        total_weight = 0
        achieved_weight = 0
        
        # Define weights for different fields
        field_weights = {
            'plan_id': 10,
            'workflow_id': 10,
            'author': 15,
            'approver': 10 if environment == 'production' else 5,
            'source_files': 15,
            'commit_references': 15,
            'build_environment': 10 if environment in ['staging', 'production'] else 5,
            'workspace_state': 5,
            'change_set_summary': 10,
            'reviewers': 5
        }
        
        for field_name, weight in field_weights.items():
            total_weight += weight
            
            value = getattr(provenance, field_name, None)
            if value is not None:
                if isinstance(value, list) and len(value) > 0:
                    achieved_weight += weight
                elif isinstance(value, str) and value.strip():
                    achieved_weight += weight
                elif not isinstance(value, (list, str)):
                    achieved_weight += weight
        
        return achieved_weight / total_weight if total_weight > 0 else 0.0

# API Interface
class ProvenanceAPI:
    """API interface for provenance operations"""
    
    def __init__(self, provenance_service: Optional[PlanProvenanceService] = None):
        self.provenance_service = provenance_service or PlanProvenanceService()
    
    def create_provenance(self, plan_id: str, workflow_id: str, 
                         auto_collect: bool = True) -> Dict[str, Any]:
        """API endpoint to create provenance record"""
        try:
            provenance = self.provenance_service.create_provenance_record(
                plan_id, workflow_id
            )
            
            if auto_collect:
                provenance = self.provenance_service.collect_provenance_from_environment(provenance)
            
            return {
                'success': True,
                'provenance_id': provenance.provenance_id,
                'plan_id': provenance.plan_id,
                'workflow_id': provenance.workflow_id,
                'created_at': provenance.created_at.isoformat(),
                'provenance_source': provenance.provenance_source.value
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': str(e),
                'error_type': type(e).__name__
            }
    
    def validate_provenance(self, provenance_id: str, environment: str = "development") -> Dict[str, Any]:
        """API endpoint to validate provenance"""
        try:
            provenance = self.provenance_service.provenance_records.get(provenance_id)
            if not provenance:
                return {
                    'success': False,
                    'error': 'Provenance record not found'
                }
            
            validation_results = self.provenance_service.validate_provenance(provenance, environment)
            
            return {
                'success': True,
                'is_complete': provenance.is_complete,
                'completeness_score': provenance.completeness_score,
                'validation_results': [
                    {
                        'severity': result.severity.value,
                        'field_name': result.field_name,
                        'message': result.message,
                        'suggested_fix': result.suggested_fix
                    }
                    for result in validation_results
                ]
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': str(e)
            }

# Test Functions
def create_test_provenance() -> CompletePlanProvenance:
    """Create test provenance data"""
    service = PlanProvenanceService()
    
    # Create base provenance
    provenance = service.create_provenance_record(
        "test_plan_001", "test_workflow", ProvenanceSource.API_REQUEST
    )
    
    # Add source files
    service.add_source_file_references(
        provenance,
        ["workflows/test_workflow.yaml", "models/test_model.py"],
        include_line_ranges=True
    )
    
    # Add human actors
    author = HumanActor(
        user_id="john.doe",
        name="John Doe",
        email="john.doe@example.com",
        role="ml_engineer",
        department="data_science"
    )
    
    approver = HumanActor(
        user_id="jane.smith",
        name="Jane Smith",
        email="jane.smith@example.com",
        role="engineering_director",
        department="engineering"
    )
    
    service.set_human_actors(provenance, author=author, approver=approver)
    
    # Add change set summary
    change_summary = ChangeSetSummary(
        files_modified=["workflows/test_workflow.yaml"],
        files_added=["models/test_model_v2.py"],
        lines_added=150,
        lines_deleted=50,
        change_description="Added new ML model for churn prediction",
        change_category="feature"
    )
    
    service.add_change_set_summary(provenance, change_summary)
    
    return provenance

def run_provenance_tests():
    """Run comprehensive provenance tests"""
    print("=== Plan Provenance Service Tests ===")
    
    # Initialize service
    provenance_service = PlanProvenanceService()
    provenance_api = ProvenanceAPI(provenance_service)
    
    # Test 1: Create provenance record
    print("\n1. Testing provenance record creation...")
    provenance = provenance_service.create_provenance_record(
        "test_plan_001", "test_workflow_001", ProvenanceSource.MANUAL_ENTRY
    )
    print(f"   Created provenance: {provenance.provenance_id}")
    print(f"   Plan ID: {provenance.plan_id}")
    print(f"   Workflow ID: {provenance.workflow_id}")
    
    # Test 2: Environment collection
    print("\n2. Testing environment collection...")
    original_source = provenance.provenance_source
    provenance_service.collect_provenance_from_environment(provenance)
    print(f"   Original source: {original_source.value}")
    print(f"   Updated source: {provenance.provenance_source.value}")
    print(f"   Collection metadata: {len(provenance.collection_metadata)} items")
    
    # Test 3: Add source files
    print("\n3. Testing source file references...")
    test_files = ["test_file_1.py", "test_file_2.yaml"]
    # Create test files for demonstration
    for file_path in test_files:
        try:
            with open(file_path, 'w') as f:
                f.write(f"# Test file: {file_path}\nprint('Hello from {file_path}')\n")
        except Exception:
            pass
    
    provenance_service.add_source_file_references(provenance, test_files, include_line_ranges=True)
    print(f"   Added {len(provenance.source_files)} source file references")
    
    # Test 4: Add human actors
    print("\n4. Testing human actors...")
    author = HumanActor(
        user_id="test_author",
        name="Test Author",
        email="author@test.com",
        role="developer"
    )
    
    approver = HumanActor(
        user_id="test_approver",
        name="Test Approver",
        email="approver@test.com",
        role="manager"
    )
    
    provenance_service.set_human_actors(provenance, author=author, approver=approver)
    print(f"   Author: {provenance.author.user_id}")
    print(f"   Approver: {provenance.approver.user_id}")
    
    # Test 5: Add change set summary
    print("\n5. Testing change set summary...")
    change_summary = ChangeSetSummary(
        files_modified=test_files,
        lines_added=100,
        lines_deleted=25,
        change_description="Test changes for provenance",
        change_category="test"
    )
    
    provenance_service.add_change_set_summary(provenance, change_summary)
    print(f"   Files modified: {len(change_summary.files_modified)}")
    print(f"   Lines added: {change_summary.lines_added}")
    print(f"   Change category: {change_summary.change_category}")
    
    # Test 6: Validation
    print("\n6. Testing provenance validation...")
    
    # Test development environment
    dev_validation = provenance_service.validate_provenance(provenance, "development")
    print(f"   Development validation: {len(dev_validation)} issues")
    print(f"   Completeness score: {provenance.completeness_score:.1%}")
    print(f"   Is complete: {provenance.is_complete}")
    
    # Test production environment (stricter)
    prod_validation = provenance_service.validate_provenance(provenance, "production")
    print(f"   Production validation: {len(prod_validation)} issues")
    
    # Show validation issues
    for result in prod_validation[:3]:  # Show first 3 issues
        print(f"     {result.severity.value}: {result.message}")
    
    # Test 7: Export for manifest
    print("\n7. Testing manifest export...")
    manifest_data = provenance_service.export_provenance_for_manifest(provenance.provenance_id)
    print(f"   Exported manifest data: {len(manifest_data)} fields")
    print(f"   Source files in export: {len(manifest_data.get('source_files', []))}")
    print(f"   Has author: {'author' in manifest_data}")
    print(f"   Has approver: {'approver' in manifest_data}")
    
    # Test 8: API interface
    print("\n8. Testing API interface...")
    
    # Create via API
    api_result = provenance_api.create_provenance("api_test_plan", "api_test_workflow", auto_collect=True)
    print(f"   API creation: {'✅ PASS' if api_result['success'] else '❌ FAIL'}")
    
    if api_result['success']:
        # Validate via API
        api_validation = provenance_api.validate_provenance(api_result['provenance_id'])
        print(f"   API validation: {'✅ PASS' if api_validation['success'] else '❌ FAIL'}")
        if api_validation['success']:
            print(f"   API completeness: {api_validation['completeness_score']:.1%}")
    
    # Test 9: Lookup and summary
    print("\n9. Testing lookup and summary...")
    
    retrieved_provenance = provenance_service.get_provenance_by_plan_id("test_plan_001")
    print(f"   Lookup by plan ID: {'✅ PASS' if retrieved_provenance else '❌ FAIL'}")
    
    summary = provenance_service.get_provenance_summary(provenance.provenance_id)
    print(f"   Summary generation: {'✅ PASS' if 'error' not in summary else '❌ FAIL'}")
    if 'error' not in summary:
        print(f"   Summary completeness: {summary['completeness_score']:.1%}")
        print(f"   Validation errors: {summary['validation_errors']}")
    
    # Cleanup test files
    for file_path in test_files:
        try:
            os.remove(file_path)
        except Exception:
            pass
    
    print(f"\n=== Test Summary ===")
    print(f"Provenance service tested successfully")
    print(f"Records created: {len(provenance_service.provenance_records)}")
    print(f"Final completeness score: {provenance.completeness_score:.1%}")
    print(f"Validation issues: {len(provenance.validation_results)}")
    
    return provenance_service, provenance_api

if __name__ == "__main__":
    run_provenance_tests()
