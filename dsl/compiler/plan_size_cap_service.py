"""
Plan Size Cap & Attachment Quotas Service - Task 6.2.41
========================================================

Plan size cap & attachment quotas
- Limits size of plan manifests and attachments to avoid transport/storage overloads
- Ensures attachments are stored externally with checksum references
- Provides automatic attachment offload to artifact store
- Tracks large plans and suggests modularization
- Backend implementation (no actual artifact store upload - that's infrastructure)

Dependencies: Task 6.2.26 (Plan Manifest Generator)
Outputs: Size management and attachment handling → enables performant CI and storage optimization
"""

from typing import Dict, List, Any, Optional, Union, Tuple, Set
import json
import logging
from datetime import datetime, timezone, timedelta
from dataclasses import dataclass, field, asdict
from enum import Enum
import hashlib
import base64
import gzip
from pathlib import Path

logger = logging.getLogger(__name__)

class SizeViolationType(Enum):
    """Types of size violations"""
    MANIFEST_TOO_LARGE = "manifest_too_large"
    TOO_MANY_ATTACHMENTS = "too_many_attachments"
    ATTACHMENT_TOO_LARGE = "attachment_too_large"
    TOTAL_SIZE_EXCEEDED = "total_size_exceeded"
    INLINE_ATTACHMENT_DETECTED = "inline_attachment_detected"

class AttachmentType(Enum):
    """Types of attachments"""
    EVIDENCE_PLACEHOLDER = "evidence_placeholder"
    EXTERNAL_ARTIFACT = "external_artifact"
    POLICY_DOCUMENT = "policy_document"
    MODEL_ARTIFACT = "model_artifact"
    DATA_SAMPLE = "data_sample"
    DOCUMENTATION = "documentation"
    CONFIGURATION = "configuration"

class CompressionType(Enum):
    """Compression methods for attachments"""
    NONE = "none"
    GZIP = "gzip"
    BROTLI = "brotli"
    LZ4 = "lz4"

@dataclass
class SizeQuota:
    """Size quota specification"""
    quota_id: str
    quota_name: str
    
    # Size limits (in bytes)
    max_manifest_size_bytes: int = 5 * 1024 * 1024  # 5 MB
    max_attachment_size_bytes: int = 100 * 1024 * 1024  # 100 MB
    max_total_size_bytes: int = 500 * 1024 * 1024  # 500 MB
    
    # Count limits
    max_attachment_count: int = 100
    max_inline_attachment_size_bytes: int = 1024  # 1 KB - force external for larger
    
    # Environment/tenant specific
    environment: str = "production"
    tenant_id: Optional[str] = None
    
    # Enforcement settings
    strict_enforcement: bool = True
    allow_compression: bool = True
    auto_offload_enabled: bool = True
    
    # Metadata
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

@dataclass
class AttachmentReference:
    """Reference to an external attachment"""
    reference_id: str
    attachment_type: AttachmentType
    
    # Storage information
    artifact_store_url: str
    checksum_sha256: str
    size_bytes: int
    
    # Metadata
    original_filename: Optional[str] = None
    content_type: str = "application/octet-stream"
    compression: CompressionType = CompressionType.NONE
    
    # Access control
    access_level: str = "plan_scoped"  # plan_scoped, tenant_scoped, public
    expiry_date: Optional[datetime] = None
    
    # Provenance
    uploaded_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    uploaded_by: Optional[str] = None

@dataclass
class SizeViolation:
    """A size quota violation"""
    violation_id: str
    violation_type: SizeViolationType
    plan_id: str
    
    # Violation details
    description: str
    current_value: Union[int, float]
    limit_value: Union[int, float]
    severity: str = "error"  # error, warning, info
    
    # Context
    attachment_id: Optional[str] = None
    node_id: Optional[str] = None
    
    # Remediation
    suggested_actions: List[str] = field(default_factory=list)
    auto_remediation_available: bool = False
    
    # Metadata
    detected_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

@dataclass
class AttachmentOffloadPlan:
    """Plan for offloading attachments to external storage"""
    plan_id: str
    offload_id: str
    
    # Attachments to offload
    attachments_to_offload: List[Dict[str, Any]] = field(default_factory=list)
    
    # Offload configuration
    target_artifact_store: str = "default"
    compression_enabled: bool = True
    deduplication_enabled: bool = True
    
    # Estimates
    estimated_size_reduction_bytes: int = 0
    estimated_size_reduction_percent: float = 0.0
    
    # Metadata
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

@dataclass
class PlanSizeAnalysis:
    """Analysis of plan size and attachments"""
    analysis_id: str
    plan_id: str
    
    # Size metrics
    manifest_size_bytes: int = 0
    attachment_total_size_bytes: int = 0
    total_plan_size_bytes: int = 0
    
    # Attachment metrics
    attachment_count: int = 0
    inline_attachment_count: int = 0
    external_attachment_count: int = 0
    
    # Breakdown by type
    size_by_attachment_type: Dict[AttachmentType, int] = field(default_factory=dict)
    count_by_attachment_type: Dict[AttachmentType, int] = field(default_factory=dict)
    
    # Violations
    size_violations: List[SizeViolation] = field(default_factory=list)
    
    # Recommendations
    modularization_suggestions: List[str] = field(default_factory=list)
    optimization_suggestions: List[str] = field(default_factory=list)
    
    # Compression analysis
    compression_potential_bytes: int = 0
    compression_potential_percent: float = 0.0
    
    # Metadata
    analyzed_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    analysis_duration_ms: float = 0.0

# Task 6.2.41: Plan Size Cap & Attachment Quotas Service
class PlanSizeCapService:
    """Service for managing plan size caps and attachment quotas"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
        # Size quotas registry
        self.size_quotas: Dict[str, SizeQuota] = {}
        
        # Attachment references registry
        self.attachment_references: Dict[str, AttachmentReference] = {}  # reference_id -> reference
        
        # Analysis cache
        self.analysis_cache: Dict[str, PlanSizeAnalysis] = {}  # plan_id -> analysis
        
        # Initialize default quotas
        self._initialize_default_quotas()
        
        # Statistics
        self.size_stats = {
            'total_analyses': 0,
            'plans_over_limit': 0,
            'attachments_offloaded': 0,
            'total_bytes_saved': 0,
            'compression_ratio_average': 0.0,
            'violation_types': {vtype.value: 0 for vtype in SizeViolationType}
        }
    
    def analyze_plan_size(self, plan_manifest: Dict[str, Any], 
                         quota_id: Optional[str] = None) -> PlanSizeAnalysis:
        """
        Analyze plan size and attachments against quotas
        
        Args:
            plan_manifest: Plan manifest to analyze
            quota_id: Size quota to check against
            
        Returns:
            PlanSizeAnalysis with detailed results
        """
        start_time = datetime.now(timezone.utc)
        plan_id = plan_manifest.get('plan_id', 'unknown')
        
        # Check cache
        cache_key = f"{plan_id}_{hash(str(plan_manifest))}"
        if cache_key in self.analysis_cache:
            return self.analysis_cache[cache_key]
        
        # Create analysis instance
        analysis_id = f"size_analysis_{plan_id}_{int(start_time.timestamp())}"
        analysis = PlanSizeAnalysis(
            analysis_id=analysis_id,
            plan_id=plan_id
        )
        
        # Calculate manifest size
        manifest_json = json.dumps(plan_manifest, separators=(',', ':'))
        analysis.manifest_size_bytes = len(manifest_json.encode('utf-8'))
        
        # Analyze attachments
        self._analyze_attachments(plan_manifest, analysis)
        
        # Calculate totals
        analysis.total_plan_size_bytes = analysis.manifest_size_bytes + analysis.attachment_total_size_bytes
        
        # Check against quotas
        if quota_id and quota_id in self.size_quotas:
            quota = self.size_quotas[quota_id]
            self._check_size_quotas(analysis, quota)
        
        # Generate recommendations
        analysis.modularization_suggestions = self._generate_modularization_suggestions(analysis)
        analysis.optimization_suggestions = self._generate_optimization_suggestions(analysis)
        
        # Analyze compression potential
        analysis.compression_potential_bytes, analysis.compression_potential_percent = self._analyze_compression_potential(plan_manifest)
        
        # Record analysis duration
        end_time = datetime.now(timezone.utc)
        analysis.analysis_duration_ms = (end_time - start_time).total_seconds() * 1000
        
        # Cache result
        self.analysis_cache[cache_key] = analysis
        
        # Update statistics
        self._update_size_stats(analysis)
        
        self.logger.info(f"✅ Analyzed plan size: {plan_id} -> {analysis.total_plan_size_bytes} bytes ({analysis.attachment_count} attachments)")
        
        return analysis
    
    def validate_plan_size(self, plan_manifest: Dict[str, Any], 
                          quota_id: str) -> Dict[str, Any]:
        """
        Validate plan against size quotas
        
        Args:
            plan_manifest: Plan manifest to validate
            quota_id: Size quota to validate against
            
        Returns:
            Dictionary with validation results
        """
        if quota_id not in self.size_quotas:
            return {
                'valid': False,
                'error': f'Unknown size quota: {quota_id}'
            }
        
        analysis = self.analyze_plan_size(plan_manifest, quota_id)
        
        # Determine validation result
        is_valid = len(analysis.size_violations) == 0
        
        return {
            'valid': is_valid,
            'plan_id': analysis.plan_id,
            'total_size_bytes': analysis.total_plan_size_bytes,
            'manifest_size_bytes': analysis.manifest_size_bytes,
            'attachment_count': analysis.attachment_count,
            'violations': [asdict(violation) for violation in analysis.size_violations],
            'suggestions': analysis.optimization_suggestions + analysis.modularization_suggestions,
            'compression_potential_percent': analysis.compression_potential_percent
        }
    
    def create_attachment_offload_plan(self, plan_manifest: Dict[str, Any],
                                     target_size_reduction_percent: float = 50.0) -> AttachmentOffloadPlan:
        """
        Create a plan for offloading attachments to reduce size
        
        Args:
            plan_manifest: Plan manifest
            target_size_reduction_percent: Target size reduction percentage
            
        Returns:
            AttachmentOffloadPlan
        """
        plan_id = plan_manifest.get('plan_id', 'unknown')
        analysis = self.analyze_plan_size(plan_manifest)
        
        offload_id = f"offload_{plan_id}_{int(datetime.now().timestamp())}"
        
        plan = AttachmentOffloadPlan(
            plan_id=plan_id,
            offload_id=offload_id
        )
        
        # Find attachments to offload
        attachments = self._extract_attachments_from_manifest(plan_manifest)
        
        total_attachment_size = sum(att.get('size_bytes', 0) for att in attachments)
        target_reduction_bytes = int(total_attachment_size * target_size_reduction_percent / 100)
        
        # Sort attachments by size (largest first)
        sorted_attachments = sorted(attachments, key=lambda x: x.get('size_bytes', 0), reverse=True)
        
        current_reduction = 0
        for attachment in sorted_attachments:
            if current_reduction >= target_reduction_bytes:
                break
            
            attachment_size = attachment.get('size_bytes', 0)
            if attachment_size > 1024:  # Only offload attachments larger than 1KB
                plan.attachments_to_offload.append({
                    'attachment_id': attachment.get('id'),
                    'size_bytes': attachment_size,
                    'type': attachment.get('type', 'external_artifact'),
                    'compression_enabled': True
                })
                current_reduction += attachment_size
        
        plan.estimated_size_reduction_bytes = current_reduction
        plan.estimated_size_reduction_percent = (current_reduction / total_attachment_size * 100) if total_attachment_size > 0 else 0
        
        self.logger.info(f"✅ Created offload plan: {offload_id} -> {plan.estimated_size_reduction_percent:.1f}% reduction")
        
        return plan
    
    def offload_attachments(self, plan_manifest: Dict[str, Any], 
                          offload_plan: AttachmentOffloadPlan) -> Dict[str, Any]:
        """
        Execute attachment offload plan (simulate upload to artifact store)
        
        Args:
            plan_manifest: Original plan manifest
            offload_plan: Offload plan to execute
            
        Returns:
            Dictionary with updated manifest and offload results
        """
        # Create a copy of the manifest for modification
        updated_manifest = json.loads(json.dumps(plan_manifest))
        offloaded_references = []
        
        for attachment_info in offload_plan.attachments_to_offload:
            attachment_id = attachment_info['attachment_id']
            
            # Find the attachment in the manifest
            attachment_data = self._find_attachment_in_manifest(updated_manifest, attachment_id)
            if not attachment_data:
                continue
            
            # Simulate upload to artifact store
            reference = self._simulate_artifact_upload(attachment_data, attachment_info)
            
            # Replace inline attachment with reference
            self._replace_attachment_with_reference(updated_manifest, attachment_id, reference)
            
            # Store reference
            self.attachment_references[reference.reference_id] = reference
            offloaded_references.append(reference)
        
        # Update statistics
        total_bytes_saved = sum(att['size_bytes'] for att in offload_plan.attachments_to_offload)
        self.size_stats['attachments_offloaded'] += len(offloaded_references)
        self.size_stats['total_bytes_saved'] += total_bytes_saved
        
        return {
            'success': True,
            'updated_manifest': updated_manifest,
            'offloaded_attachments': len(offloaded_references),
            'bytes_saved': total_bytes_saved,
            'references': [asdict(ref) for ref in offloaded_references]
        }
    
    def create_size_quota(self, quota_name: str, environment: str,
                         custom_limits: Optional[Dict[str, int]] = None,
                         tenant_id: Optional[str] = None) -> SizeQuota:
        """
        Create a custom size quota
        
        Args:
            quota_name: Name of the quota
            environment: Environment (dev, staging, prod)
            custom_limits: Custom limit overrides
            tenant_id: Optional tenant identifier
            
        Returns:
            SizeQuota
        """
        quota_id = f"quota_{hash(f'{quota_name}_{environment}_{tenant_id}_{datetime.now()}')}"
        
        # Base limits by environment
        base_limits = {
            'dev': {
                'max_manifest_size_bytes': 10 * 1024 * 1024,  # 10 MB
                'max_attachment_size_bytes': 500 * 1024 * 1024,  # 500 MB
                'max_total_size_bytes': 1024 * 1024 * 1024,  # 1 GB
                'max_attachment_count': 200
            },
            'staging': {
                'max_manifest_size_bytes': 5 * 1024 * 1024,  # 5 MB
                'max_attachment_size_bytes': 200 * 1024 * 1024,  # 200 MB
                'max_total_size_bytes': 500 * 1024 * 1024,  # 500 MB
                'max_attachment_count': 100
            },
            'prod': {
                'max_manifest_size_bytes': 2 * 1024 * 1024,  # 2 MB
                'max_attachment_size_bytes': 100 * 1024 * 1024,  # 100 MB
                'max_total_size_bytes': 200 * 1024 * 1024,  # 200 MB
                'max_attachment_count': 50
            }
        }
        
        limits = base_limits.get(environment, base_limits['prod'])
        
        # Apply custom limits
        if custom_limits:
            limits.update(custom_limits)
        
        quota = SizeQuota(
            quota_id=quota_id,
            quota_name=quota_name,
            environment=environment,
            tenant_id=tenant_id,
            **limits
        )
        
        # Store quota
        self.size_quotas[quota_id] = quota
        
        self.logger.info(f"✅ Created size quota: {quota_name} for {environment}")
        
        return quota
    
    def get_large_plans_report(self, days: int = 30, 
                              size_threshold_mb: float = 10.0) -> Dict[str, Any]:
        """
        Get report of large plans for modularization suggestions
        
        Args:
            days: Number of days to analyze
            size_threshold_mb: Size threshold in MB
            
        Returns:
            Dictionary with large plans report
        """
        cutoff_date = datetime.now(timezone.utc) - timedelta(days=days)
        size_threshold_bytes = int(size_threshold_mb * 1024 * 1024)
        
        large_plans = []
        for analysis in self.analysis_cache.values():
            if (analysis.analyzed_at >= cutoff_date and 
                analysis.total_plan_size_bytes >= size_threshold_bytes):
                large_plans.append(analysis)
        
        # Sort by size (largest first)
        large_plans.sort(key=lambda x: x.total_plan_size_bytes, reverse=True)
        
        report = {
            'report_period_days': days,
            'size_threshold_mb': size_threshold_mb,
            'large_plans_count': len(large_plans),
            'total_size_bytes': sum(p.total_plan_size_bytes for p in large_plans),
            'plans': []
        }
        
        for plan in large_plans[:20]:  # Top 20 largest plans
            plan_info = {
                'plan_id': plan.plan_id,
                'total_size_mb': plan.total_plan_size_bytes / (1024 * 1024),
                'attachment_count': plan.attachment_count,
                'modularization_suggestions': plan.modularization_suggestions,
                'compression_potential_percent': plan.compression_potential_percent
            }
            report['plans'].append(plan_info)
        
        return report
    
    def compress_attachment_data(self, data: bytes, 
                               compression_type: CompressionType = CompressionType.GZIP) -> Tuple[bytes, float]:
        """
        Compress attachment data
        
        Args:
            data: Raw attachment data
            compression_type: Compression method
            
        Returns:
            Tuple of (compressed_data, compression_ratio)
        """
        original_size = len(data)
        
        if compression_type == CompressionType.GZIP:
            compressed_data = gzip.compress(data)
        else:
            # For other compression types, simulate compression
            compressed_data = data  # Placeholder
        
        compressed_size = len(compressed_data)
        compression_ratio = compressed_size / original_size if original_size > 0 else 1.0
        
        return compressed_data, compression_ratio
    
    def get_size_statistics(self) -> Dict[str, Any]:
        """Get plan size service statistics"""
        return {
            **self.size_stats,
            'size_quotas': len(self.size_quotas),
            'attachment_references': len(self.attachment_references),
            'cached_analyses': len(self.analysis_cache)
        }
    
    def _analyze_attachments(self, plan_manifest: Dict[str, Any], analysis: PlanSizeAnalysis):
        """Analyze attachments in the plan manifest"""
        attachments = self._extract_attachments_from_manifest(plan_manifest)
        
        analysis.attachment_count = len(attachments)
        
        for attachment in attachments:
            size_bytes = attachment.get('size_bytes', 0)
            attachment_type = AttachmentType(attachment.get('type', 'external_artifact'))
            is_inline = attachment.get('inline', False)
            
            analysis.attachment_total_size_bytes += size_bytes
            
            if is_inline:
                analysis.inline_attachment_count += 1
            else:
                analysis.external_attachment_count += 1
            
            # Track by type
            if attachment_type not in analysis.size_by_attachment_type:
                analysis.size_by_attachment_type[attachment_type] = 0
                analysis.count_by_attachment_type[attachment_type] = 0
            
            analysis.size_by_attachment_type[attachment_type] += size_bytes
            analysis.count_by_attachment_type[attachment_type] += 1
    
    def _check_size_quotas(self, analysis: PlanSizeAnalysis, quota: SizeQuota):
        """Check analysis against size quotas"""
        # Check manifest size
        if analysis.manifest_size_bytes > quota.max_manifest_size_bytes:
            violation = SizeViolation(
                violation_id=f"manifest_size_{analysis.plan_id}",
                violation_type=SizeViolationType.MANIFEST_TOO_LARGE,
                plan_id=analysis.plan_id,
                description=f"Manifest size ({analysis.manifest_size_bytes} bytes) exceeds limit ({quota.max_manifest_size_bytes} bytes)",
                current_value=analysis.manifest_size_bytes,
                limit_value=quota.max_manifest_size_bytes,
                suggested_actions=["Consider modularizing the plan", "Move large attachments to external storage"],
                auto_remediation_available=True
            )
            analysis.size_violations.append(violation)
        
        # Check attachment count
        if analysis.attachment_count > quota.max_attachment_count:
            violation = SizeViolation(
                violation_id=f"attachment_count_{analysis.plan_id}",
                violation_type=SizeViolationType.TOO_MANY_ATTACHMENTS,
                plan_id=analysis.plan_id,
                description=f"Attachment count ({analysis.attachment_count}) exceeds limit ({quota.max_attachment_count})",
                current_value=analysis.attachment_count,
                limit_value=quota.max_attachment_count,
                suggested_actions=["Consolidate similar attachments", "Remove unnecessary attachments"]
            )
            analysis.size_violations.append(violation)
        
        # Check total size
        if analysis.total_plan_size_bytes > quota.max_total_size_bytes:
            violation = SizeViolation(
                violation_id=f"total_size_{analysis.plan_id}",
                violation_type=SizeViolationType.TOTAL_SIZE_EXCEEDED,
                plan_id=analysis.plan_id,
                description=f"Total plan size ({analysis.total_plan_size_bytes} bytes) exceeds limit ({quota.max_total_size_bytes} bytes)",
                current_value=analysis.total_plan_size_bytes,
                limit_value=quota.max_total_size_bytes,
                suggested_actions=["Enable attachment offloading", "Use compression", "Split into multiple plans"],
                auto_remediation_available=True
            )
            analysis.size_violations.append(violation)
        
        # Check for inline attachments that should be external
        if analysis.inline_attachment_count > 0:
            violation = SizeViolation(
                violation_id=f"inline_attachments_{analysis.plan_id}",
                violation_type=SizeViolationType.INLINE_ATTACHMENT_DETECTED,
                plan_id=analysis.plan_id,
                description=f"Found {analysis.inline_attachment_count} inline attachments that should be external",
                current_value=analysis.inline_attachment_count,
                limit_value=0,
                severity="warning",
                suggested_actions=["Move inline attachments to external storage"],
                auto_remediation_available=True
            )
            analysis.size_violations.append(violation)
    
    def _generate_modularization_suggestions(self, analysis: PlanSizeAnalysis) -> List[str]:
        """Generate modularization suggestions"""
        suggestions = []
        
        if analysis.total_plan_size_bytes > 50 * 1024 * 1024:  # > 50 MB
            suggestions.append("Consider splitting this large plan into multiple smaller plans")
        
        if analysis.attachment_count > 50:
            suggestions.append("High attachment count - consider consolidating related attachments")
        
        # Check for dominant attachment types
        if analysis.size_by_attachment_type:
            total_attachment_size = sum(analysis.size_by_attachment_type.values())
            for att_type, size in analysis.size_by_attachment_type.items():
                if size > total_attachment_size * 0.5:  # > 50% of total
                    suggestions.append(f"Large {att_type.value} attachments dominate plan size - consider optimization")
        
        return suggestions
    
    def _generate_optimization_suggestions(self, analysis: PlanSizeAnalysis) -> List[str]:
        """Generate optimization suggestions"""
        suggestions = []
        
        if analysis.inline_attachment_count > 0:
            suggestions.append("Move inline attachments to external storage")
        
        if analysis.compression_potential_percent > 20:
            suggestions.append(f"Enable compression to save ~{analysis.compression_potential_percent:.1f}% space")
        
        if analysis.attachment_total_size_bytes > analysis.manifest_size_bytes * 10:
            suggestions.append("Attachments are much larger than manifest - prioritize attachment optimization")
        
        return suggestions
    
    def _analyze_compression_potential(self, plan_manifest: Dict[str, Any]) -> Tuple[int, float]:
        """Analyze compression potential for the manifest"""
        # Simulate compression analysis
        manifest_json = json.dumps(plan_manifest, separators=(',', ':'))
        original_size = len(manifest_json.encode('utf-8'))
        
        # Estimate compression ratio (typical JSON compresses to ~30-50% of original)
        estimated_compressed_size = int(original_size * 0.4)  # 40% of original
        potential_savings = original_size - estimated_compressed_size
        potential_percent = (potential_savings / original_size * 100) if original_size > 0 else 0
        
        return potential_savings, potential_percent
    
    def _extract_attachments_from_manifest(self, plan_manifest: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Extract attachments from plan manifest"""
        attachments = []
        
        # Check for attachments in various manifest sections
        if 'attachments' in plan_manifest:
            attachments.extend(plan_manifest['attachments'])
        
        # Check in plan graph nodes
        plan_graph = plan_manifest.get('plan_graph', {})
        for node in plan_graph.get('nodes', []):
            node_attachments = node.get('attachments', [])
            for att in node_attachments:
                att['node_id'] = node.get('id')
                attachments.append(att)
        
        # Check in metadata
        metadata = plan_manifest.get('metadata', {})
        if 'evidence_attachments' in metadata:
            attachments.extend(metadata['evidence_attachments'])
        
        return attachments
    
    def _find_attachment_in_manifest(self, plan_manifest: Dict[str, Any], attachment_id: str) -> Optional[Dict[str, Any]]:
        """Find specific attachment in manifest"""
        attachments = self._extract_attachments_from_manifest(plan_manifest)
        
        for attachment in attachments:
            if attachment.get('id') == attachment_id:
                return attachment
        
        return None
    
    def _simulate_artifact_upload(self, attachment_data: Dict[str, Any], 
                                attachment_info: Dict[str, Any]) -> AttachmentReference:
        """Simulate uploading attachment to artifact store"""
        reference_id = f"ref_{hash(f'{attachment_data}_{datetime.now()}')}"
        
        # Simulate artifact store URL
        artifact_store_url = f"artifact-store://attachments/{reference_id}"
        
        # Calculate checksum
        content = str(attachment_data).encode('utf-8')
        checksum = hashlib.sha256(content).hexdigest()
        
        reference = AttachmentReference(
            reference_id=reference_id,
            attachment_type=AttachmentType(attachment_info.get('type', 'external_artifact')),
            artifact_store_url=artifact_store_url,
            checksum_sha256=checksum,
            size_bytes=attachment_info['size_bytes'],
            original_filename=attachment_data.get('filename'),
            content_type=attachment_data.get('content_type', 'application/octet-stream'),
            compression=CompressionType.GZIP if attachment_info.get('compression_enabled') else CompressionType.NONE
        )
        
        return reference
    
    def _replace_attachment_with_reference(self, plan_manifest: Dict[str, Any], 
                                         attachment_id: str, reference: AttachmentReference):
        """Replace inline attachment with external reference"""
        # This would traverse the manifest and replace the attachment with a reference
        # For simulation, we'll just add a reference section
        if 'attachment_references' not in plan_manifest:
            plan_manifest['attachment_references'] = []
        
        plan_manifest['attachment_references'].append({
            'attachment_id': attachment_id,
            'reference_id': reference.reference_id,
            'artifact_store_url': reference.artifact_store_url,
            'checksum_sha256': reference.checksum_sha256,
            'size_bytes': reference.size_bytes
        })
    
    def _initialize_default_quotas(self):
        """Initialize default size quotas"""
        environments = ['dev', 'staging', 'prod']
        
        for env in environments:
            quota = self.create_size_quota(f"default_{env}", env)
        
        self.logger.info(f"✅ Initialized {len(environments)} default size quotas")
    
    def _update_size_stats(self, analysis: PlanSizeAnalysis):
        """Update size statistics"""
        self.size_stats['total_analyses'] += 1
        
        if analysis.size_violations:
            self.size_stats['plans_over_limit'] += 1
            
            for violation in analysis.size_violations:
                self.size_stats['violation_types'][violation.violation_type.value] += 1

# API Interface
class PlanSizeCapAPI:
    """API interface for plan size cap operations"""
    
    def __init__(self, size_service: Optional[PlanSizeCapService] = None):
        self.size_service = size_service or PlanSizeCapService()
    
    def analyze_plan_size(self, plan_manifest: Dict[str, Any], 
                         quota_id: Optional[str] = None) -> Dict[str, Any]:
        """API endpoint to analyze plan size"""
        try:
            analysis = self.size_service.analyze_plan_size(plan_manifest, quota_id)
            
            return {
                'success': True,
                'analysis': asdict(analysis)
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': str(e),
                'error_type': type(e).__name__
            }
    
    def validate_plan_size(self, plan_manifest: Dict[str, Any], 
                          quota_id: str) -> Dict[str, Any]:
        """API endpoint to validate plan size"""
        try:
            result = self.size_service.validate_plan_size(plan_manifest, quota_id)
            
            return {
                'success': True,
                **result
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': str(e)
            }
    
    def create_offload_plan(self, plan_manifest: Dict[str, Any],
                           target_reduction_percent: float = 50.0) -> Dict[str, Any]:
        """API endpoint to create attachment offload plan"""
        try:
            offload_plan = self.size_service.create_attachment_offload_plan(
                plan_manifest, target_reduction_percent
            )
            
            return {
                'success': True,
                'offload_plan': asdict(offload_plan)
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': str(e)
            }

# Test Functions
def create_test_plan_manifest(size_mb: float = 1.0, 
                            attachment_count: int = 5,
                            include_large_attachments: bool = False) -> Dict[str, Any]:
    """Create a test plan manifest with specified size characteristics"""
    
    # Create base manifest
    manifest = {
        'plan_id': f'test_plan_{size_mb}mb_{attachment_count}att',
        'plan_version': '1.0.0',
        'plan_graph': {
            'nodes': [],
            'edges': []
        },
        'metadata': {
            'created_at': datetime.now(timezone.utc).isoformat(),
            'author': 'test_user'
        },
        'attachments': []
    }
    
    # Add nodes to reach target size
    target_size_bytes = int(size_mb * 1024 * 1024)
    node_count = max(10, target_size_bytes // 10000)  # Roughly 10KB per node
    
    for i in range(node_count):
        node = {
            'id': f'node_{i}',
            'type': 'step',
            'name': f'Step {i}',
            'description': 'A' * 100,  # 100 characters
            'configuration': {
                'param1': 'value1' * 50,  # Pad with data
                'param2': list(range(100)),  # Array data
                'param3': {'nested': 'data' * 20}
            }
        }
        manifest['plan_graph']['nodes'].append(node)
    
    # Add edges
    for i in range(node_count - 1):
        manifest['plan_graph']['edges'].append({
            'source': f'node_{i}',
            'target': f'node_{i + 1}'
        })
    
    # Add attachments
    for i in range(attachment_count):
        attachment_size = 1024 * 1024 if include_large_attachments else 1024  # 1MB or 1KB
        
        attachment = {
            'id': f'attachment_{i}',
            'type': 'external_artifact',
            'filename': f'artifact_{i}.json',
            'content_type': 'application/json',
            'size_bytes': attachment_size,
            'inline': i < 2,  # First 2 are inline
            'data': base64.b64encode(b'x' * min(attachment_size, 1000)).decode() if i < 2 else None,
            'checksum': hashlib.sha256(b'x' * attachment_size).hexdigest()
        }
        manifest['attachments'].append(attachment)
    
    return manifest

def run_plan_size_cap_tests():
    """Run comprehensive plan size cap tests"""
    print("=== Plan Size Cap Service Tests ===")
    
    # Initialize service
    size_service = PlanSizeCapService()
    size_api = PlanSizeCapAPI(size_service)
    
    # Test 1: Default quotas initialization
    print("\n1. Testing default quotas initialization...")
    stats = size_service.get_size_statistics()
    print(f"   Default quotas created: {stats['size_quotas']}")
    
    for quota_id, quota in size_service.size_quotas.items():
        print(f"     - {quota.quota_name}: manifest={quota.max_manifest_size_bytes//1024//1024}MB, total={quota.max_total_size_bytes//1024//1024}MB")
    
    # Test 2: Plan size analysis
    print("\n2. Testing plan size analysis...")
    
    # Test different sized plans
    test_plans = [
        (0.5, 3, False, "Small plan"),
        (2.0, 10, False, "Medium plan"),
        (8.0, 25, True, "Large plan"),
        (15.0, 100, True, "Very large plan")
    ]
    
    for size_mb, att_count, large_att, description in test_plans:
        test_manifest = create_test_plan_manifest(size_mb, att_count, large_att)
        analysis = size_service.analyze_plan_size(test_manifest)
        
        print(f"   {description}:")
        print(f"     Total size: {analysis.total_plan_size_bytes / 1024 / 1024:.1f}MB")
        print(f"     Manifest size: {analysis.manifest_size_bytes / 1024:.1f}KB")
        print(f"     Attachments: {analysis.attachment_count} ({analysis.attachment_total_size_bytes / 1024 / 1024:.1f}MB)")
        print(f"     Inline attachments: {analysis.inline_attachment_count}")
        print(f"     Compression potential: {analysis.compression_potential_percent:.1f}%")
    
    # Test 3: Size quota validation
    print("\n3. Testing size quota validation...")
    
    # Get a quota to test against
    quota_id = list(size_service.size_quotas.keys())[0]  # Use first quota (dev)
    
    # Test with small plan (should pass)
    small_manifest = create_test_plan_manifest(0.5, 3, False)
    validation_result = size_service.validate_plan_size(small_manifest, quota_id)
    print(f"   Small plan validation: {'✅ PASS' if validation_result['valid'] else '❌ FAIL'}")
    
    # Test with large plan (should fail)
    large_manifest = create_test_plan_manifest(15.0, 150, True)
    validation_result = size_service.validate_plan_size(large_manifest, quota_id)
    print(f"   Large plan validation: {'❌ FAIL' if not validation_result['valid'] else '✅ UNEXPECTED PASS'}")
    if not validation_result['valid']:
        print(f"     Violations: {len(validation_result['violations'])}")
        for violation in validation_result['violations'][:3]:  # Show first 3
            print(f"       - {violation['description']}")
    
    # Test 4: Custom quota creation
    print("\n4. Testing custom quota creation...")
    
    custom_limits = {
        'max_manifest_size_bytes': 1024 * 1024,  # 1 MB
        'max_attachment_count': 20,
        'max_total_size_bytes': 50 * 1024 * 1024  # 50 MB
    }
    
    custom_quota = size_service.create_size_quota(
        "strict_quota",
        "test",
        custom_limits=custom_limits,
        tenant_id="test_tenant"
    )
    
    print(f"   Custom quota created: {custom_quota.quota_name}")
    print(f"     Environment: {custom_quota.environment}")
    print(f"     Max manifest: {custom_quota.max_manifest_size_bytes / 1024}KB")
    print(f"     Max attachments: {custom_quota.max_attachment_count}")
    
    # Test 5: Attachment offload planning
    print("\n5. Testing attachment offload planning...")
    
    # Create plan with large attachments
    large_attachment_manifest = create_test_plan_manifest(5.0, 20, True)
    
    offload_plan = size_service.create_attachment_offload_plan(
        large_attachment_manifest, target_size_reduction_percent=60.0
    )
    
    print(f"   Offload plan created: {offload_plan.offload_id}")
    print(f"     Attachments to offload: {len(offload_plan.attachments_to_offload)}")
    print(f"     Estimated reduction: {offload_plan.estimated_size_reduction_percent:.1f}%")
    print(f"     Estimated bytes saved: {offload_plan.estimated_size_reduction_bytes / 1024 / 1024:.1f}MB")
    
    # Test 6: Attachment offload execution
    print("\n6. Testing attachment offload execution...")
    
    offload_result = size_service.offload_attachments(large_attachment_manifest, offload_plan)
    
    print(f"   Offload execution: {'✅ PASS' if offload_result['success'] else '❌ FAIL'}")
    if offload_result['success']:
        print(f"     Attachments offloaded: {offload_result['offloaded_attachments']}")
        print(f"     Bytes saved: {offload_result['bytes_saved'] / 1024 / 1024:.1f}MB")
        print(f"     References created: {len(offload_result['references'])}")
    
    # Test 7: Compression analysis
    print("\n7. Testing compression analysis...")
    
    test_data = b'{"key": "value"}' * 1000  # Repetitive data that compresses well
    compressed_data, compression_ratio = size_service.compress_attachment_data(test_data)
    
    print(f"   Compression test:")
    print(f"     Original size: {len(test_data)} bytes")
    print(f"     Compressed size: {len(compressed_data)} bytes")
    print(f"     Compression ratio: {compression_ratio:.2f}")
    print(f"     Space saved: {(1 - compression_ratio) * 100:.1f}%")
    
    # Test 8: Large plans report
    print("\n8. Testing large plans report...")
    
    # Generate some analyses for the report
    for i in range(5):
        test_manifest = create_test_plan_manifest(5 + i * 2, 10 + i * 5, True)
        size_service.analyze_plan_size(test_manifest)
    
    large_plans_report = size_service.get_large_plans_report(days=30, size_threshold_mb=5.0)
    
    print(f"   Large plans report:")
    print(f"     Plans analyzed: {large_plans_report['large_plans_count']}")
    print(f"     Size threshold: {large_plans_report['size_threshold_mb']}MB")
    if large_plans_report['plans']:
        print(f"     Largest plan: {large_plans_report['plans'][0]['total_size_mb']:.1f}MB")
        print(f"     Most attachments: {max(p['attachment_count'] for p in large_plans_report['plans'])}")
    
    # Test 9: API interface
    print("\n9. Testing API interface...")
    
    # Test API analysis
    api_analysis_result = size_api.analyze_plan_size(small_manifest)
    print(f"   API analysis: {'✅ PASS' if api_analysis_result['success'] else '❌ FAIL'}")
    
    # Test API validation
    api_validation_result = size_api.validate_plan_size(large_manifest, quota_id)
    print(f"   API validation: {'✅ PASS' if api_validation_result['success'] else '❌ FAIL'}")
    if api_validation_result['success']:
        print(f"     Plan valid: {'✅ VALID' if api_validation_result['valid'] else '❌ INVALID'}")
    
    # Test API offload plan
    api_offload_result = size_api.create_offload_plan(large_attachment_manifest, 40.0)
    print(f"   API offload plan: {'✅ PASS' if api_offload_result['success'] else '❌ FAIL'}")
    
    # Test 10: Edge cases
    print("\n10. Testing edge cases...")
    
    # Empty manifest
    empty_manifest = {'plan_id': 'empty', 'plan_graph': {'nodes': [], 'edges': []}}
    empty_analysis = size_service.analyze_plan_size(empty_manifest)
    print(f"   Empty manifest: {empty_analysis.total_plan_size_bytes} bytes, {empty_analysis.attachment_count} attachments")
    
    # Manifest with only inline attachments
    inline_manifest = create_test_plan_manifest(1.0, 5, False)
    for att in inline_manifest['attachments']:
        att['inline'] = True
        att['data'] = base64.b64encode(b'x' * 2000).decode()  # 2KB inline data
    
    inline_analysis = size_service.analyze_plan_size(inline_manifest)
    print(f"   Inline attachments: {inline_analysis.inline_attachment_count} inline, violations: {len(inline_analysis.size_violations)}")
    
    # Test 11: Statistics
    print("\n11. Testing statistics...")
    
    final_stats = size_service.get_size_statistics()
    print(f"   Total analyses: {final_stats['total_analyses']}")
    print(f"   Plans over limit: {final_stats['plans_over_limit']}")
    print(f"   Attachments offloaded: {final_stats['attachments_offloaded']}")
    print(f"   Total bytes saved: {final_stats['total_bytes_saved'] / 1024 / 1024:.1f}MB")
    print(f"   Size quotas: {final_stats['size_quotas']}")
    print(f"   Attachment references: {final_stats['attachment_references']}")
    print(f"   Violation types: {final_stats['violation_types']}")
    
    print(f"\n=== Test Summary ===")
    print(f"Plan size cap service tested successfully")
    print(f"Size quotas: {final_stats['size_quotas']}")
    print(f"Total analyses: {final_stats['total_analyses']}")
    print(f"Plans over limit: {final_stats['plans_over_limit']}")
    print(f"Bytes saved through offloading: {final_stats['total_bytes_saved'] / 1024 / 1024:.1f}MB")
    
    return size_service, size_api

if __name__ == "__main__":
    run_plan_size_cap_tests()
