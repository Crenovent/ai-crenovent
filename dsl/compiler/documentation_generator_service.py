"""
Documentation Generator Service - Task 6.2.43
==============================================

Docs generator (plan → human doc with diagrams)
- Auto-generates documentation for compiled plans
- Creates human-readable summaries, node descriptions, policy impacts
- Includes SLA & cost summaries and diagrams
- Templates per persona (CRO, Compliance, Dev) with different detail levels
- Backend implementation (no actual PDF generation - that's infrastructure)

Dependencies: Task 6.2.42 (IR Pretty-Printer), Task 6.2.44 (Graph Visualizer)
Outputs: Plan documentation → enables stakeholder reviews and compliance audits
"""

from typing import Dict, List, Any, Optional, Union, Tuple, Set
import json
import logging
from datetime import datetime, timezone
from dataclasses import dataclass, field, asdict
from enum import Enum
import re
from collections import defaultdict
import statistics

logger = logging.getLogger(__name__)

class DocumentFormat(Enum):
    """Document output formats"""
    MARKDOWN = "markdown"
    HTML = "html"
    PDF = "pdf"
    DOCX = "docx"
    JSON = "json"

class PersonaType(Enum):
    """Target personas for documentation"""
    DEVELOPER = "developer"
    CRO = "cro"                    # Chief Revenue Officer
    COMPLIANCE = "compliance"
    AUDITOR = "auditor"
    STAKEHOLDER = "stakeholder"
    TECHNICAL_REVIEWER = "technical_reviewer"

class RiskTier(Enum):
    """Risk tiers for plans"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

@dataclass
class ExecutiveSummary:
    """Executive summary of the plan"""
    plan_purpose: str
    business_impact: str
    risk_tier: RiskTier
    
    # Key metrics
    total_nodes: int = 0
    ml_nodes: int = 0
    external_integrations: int = 0
    estimated_cost_monthly: float = 0.0
    
    # Compliance
    compliance_requirements: List[str] = field(default_factory=list)
    data_residency_regions: List[str] = field(default_factory=list)
    
    # Provenance
    author: Optional[str] = None
    approver: Optional[str] = None
    created_date: Optional[str] = None
    last_modified: Optional[str] = None

@dataclass
class NodeDocumentation:
    """Documentation for a single node"""
    node_id: str
    node_type: str
    display_name: str
    description: str
    
    # Technical details
    inputs: List[Dict[str, str]] = field(default_factory=list)
    outputs: List[Dict[str, str]] = field(default_factory=list)
    
    # Business context
    business_purpose: str = ""
    data_processing_description: str = ""
    
    # Policies and compliance
    policies: Dict[str, Any] = field(default_factory=dict)
    compliance_notes: List[str] = field(default_factory=list)
    
    # Performance and cost
    sla_requirements: Dict[str, str] = field(default_factory=dict)
    cost_class: str = "unknown"
    estimated_execution_time: Optional[str] = None
    
    # ML-specific (if applicable)
    model_info: Optional[Dict[str, Any]] = None
    trust_budget: Optional[Dict[str, Any]] = None
    fallback_strategy: Optional[str] = None
    explainability_config: Optional[Dict[str, Any]] = None

@dataclass
class PolicyImpactSummary:
    """Summary of policy impacts"""
    # Residency and compliance
    data_residency_constraints: List[str] = field(default_factory=list)
    regulatory_compliance: List[str] = field(default_factory=list)
    
    # Security and governance
    access_controls: List[str] = field(default_factory=list)
    audit_requirements: List[str] = field(default_factory=list)
    
    # Risk management
    risk_mitigation_measures: List[str] = field(default_factory=list)
    fallback_coverage: float = 0.0  # Percentage of nodes with fallbacks
    
    # Trust and ML governance
    ml_governance_policies: List[str] = field(default_factory=list)
    trust_thresholds: Dict[str, float] = field(default_factory=dict)

@dataclass
class CostSummary:
    """Cost analysis summary"""
    # Cost breakdown
    estimated_monthly_cost: float = 0.0
    cost_by_category: Dict[str, float] = field(default_factory=dict)
    cost_by_node_type: Dict[str, float] = field(default_factory=dict)
    
    # Resource usage
    estimated_compute_hours: float = 0.0
    estimated_storage_gb: float = 0.0
    estimated_network_gb: float = 0.0
    
    # ML-specific costs
    ml_inference_costs: float = 0.0
    model_training_costs: float = 0.0
    
    # Cost optimization suggestions
    optimization_opportunities: List[str] = field(default_factory=list)

@dataclass
class SLASummary:
    """SLA requirements summary"""
    # Performance targets
    target_availability: str = "99.9%"
    target_response_time: str = "< 100ms"
    target_throughput: str = "1000 req/min"
    
    # Recovery objectives
    rto_target: str = "< 5 minutes"  # Recovery Time Objective
    rpo_target: str = "< 1 minute"   # Recovery Point Objective
    
    # Monitoring and alerting
    monitoring_requirements: List[str] = field(default_factory=list)
    alert_thresholds: Dict[str, str] = field(default_factory=dict)
    
    # Dependencies
    critical_dependencies: List[str] = field(default_factory=list)
    external_service_slas: Dict[str, str] = field(default_factory=dict)

@dataclass
class EvidenceCheckpointSummary:
    """Evidence and audit checkpoint summary"""
    # Checkpoint configuration
    checkpoint_frequency: str = "per_node"
    data_retention_period: str = "7 years"
    
    # Evidence types
    evidence_types_collected: List[str] = field(default_factory=list)
    audit_trail_completeness: str = "full"
    
    # Compliance alignment
    regulatory_evidence_mapping: Dict[str, List[str]] = field(default_factory=dict)
    
    # Storage and access
    evidence_storage_location: str = "secure_vault"
    access_control_policy: str = "need_to_know"

@dataclass
class PlanDocumentation:
    """Complete plan documentation"""
    doc_id: str
    plan_id: str
    plan_version: str
    
    # Core sections
    executive_summary: ExecutiveSummary
    node_documentation: List[NodeDocumentation] = field(default_factory=list)
    policy_impact_summary: PolicyImpactSummary = field(default_factory=PolicyImpactSummary)
    cost_summary: CostSummary = field(default_factory=CostSummary)
    sla_summary: SLASummary = field(default_factory=SLASummary)
    evidence_summary: EvidenceCheckpointSummary = field(default_factory=EvidenceCheckpointSummary)
    
    # Diagrams and visualizations
    topology_diagram: Optional[str] = None
    fallback_diagram: Optional[str] = None
    data_flow_diagram: Optional[str] = None
    
    # Persona-specific content
    target_persona: PersonaType = PersonaType.STAKEHOLDER
    detail_level: str = "standard"  # minimal, standard, detailed, comprehensive
    
    # Document metadata
    generated_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    generated_by: str = "RBIA Documentation Generator"
    document_version: str = "1.0"
    
    # Review and approval
    review_status: str = "draft"  # draft, under_review, approved
    reviewers: List[str] = field(default_factory=list)
    approval_date: Optional[datetime] = None

# Task 6.2.43: Documentation Generator Service
class DocumentationGeneratorService:
    """Service for generating plan documentation"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
        # Document templates by persona
        self.persona_templates = self._initialize_persona_templates()
        
        # Cost estimation models
        self.cost_models = self._initialize_cost_models()
        
        # SLA templates
        self.sla_templates = self._initialize_sla_templates()
        
        # Generated documents cache
        self.document_cache: Dict[str, PlanDocumentation] = {}
        
        # Statistics
        self.doc_stats = {
            'total_documents_generated': 0,
            'documents_by_persona': {persona.value: 0 for persona in PersonaType},
            'documents_by_format': {fmt.value: 0 for fmt in DocumentFormat},
            'average_generation_time_ms': 0.0,
            'cache_hits': 0,
            'cache_misses': 0
        }
    
    def generate_plan_documentation(self, plan_manifest: Dict[str, Any],
                                  target_persona: PersonaType = PersonaType.STAKEHOLDER,
                                  detail_level: str = "standard") -> PlanDocumentation:
        """
        Generate comprehensive documentation for a plan
        
        Args:
            plan_manifest: Plan manifest to document
            target_persona: Target audience for the documentation
            detail_level: Level of detail (minimal, standard, detailed, comprehensive)
            
        Returns:
            PlanDocumentation with complete documentation
        """
        start_time = datetime.now(timezone.utc)
        
        plan_id = plan_manifest.get('plan_id', 'unknown')
        plan_version = plan_manifest.get('version', '1.0')
        
        # Check cache
        cache_key = f"{plan_id}_{plan_version}_{target_persona.value}_{detail_level}"
        if cache_key in self.document_cache:
            self.doc_stats['cache_hits'] += 1
            return self.document_cache[cache_key]
        
        self.doc_stats['cache_misses'] += 1
        
        # Create documentation instance
        doc_id = f"doc_{plan_id}_{target_persona.value}_{int(start_time.timestamp())}"
        
        documentation = PlanDocumentation(
            doc_id=doc_id,
            plan_id=plan_id,
            plan_version=plan_version,
            target_persona=target_persona,
            detail_level=detail_level,
            executive_summary=ExecutiveSummary(
                plan_purpose="Auto-generated documentation",
                business_impact="To be determined",
                risk_tier=RiskTier.MEDIUM
            )
        )
        
        # Generate executive summary
        documentation.executive_summary = self._generate_executive_summary(plan_manifest)
        
        # Generate node documentation
        documentation.node_documentation = self._generate_node_documentation(plan_manifest, target_persona, detail_level)
        
        # Generate policy impact summary
        documentation.policy_impact_summary = self._generate_policy_impact_summary(plan_manifest)
        
        # Generate cost summary
        documentation.cost_summary = self._generate_cost_summary(plan_manifest)
        
        # Generate SLA summary
        documentation.sla_summary = self._generate_sla_summary(plan_manifest)
        
        # Generate evidence checkpoint summary
        documentation.evidence_summary = self._generate_evidence_summary(plan_manifest)
        
        # Generate diagram placeholders (actual diagram generation would be handled by Task 6.2.44)
        documentation.topology_diagram = self._generate_topology_diagram_placeholder(plan_manifest)
        documentation.fallback_diagram = self._generate_fallback_diagram_placeholder(plan_manifest)
        documentation.data_flow_diagram = self._generate_data_flow_diagram_placeholder(plan_manifest)
        
        # Cache result
        self.document_cache[cache_key] = documentation
        
        # Update statistics
        end_time = datetime.now(timezone.utc)
        generation_time = (end_time - start_time).total_seconds() * 1000
        
        self.doc_stats['total_documents_generated'] += 1
        self.doc_stats['documents_by_persona'][target_persona.value] += 1
        
        # Update average generation time
        total_docs = self.doc_stats['total_documents_generated']
        current_avg = self.doc_stats['average_generation_time_ms']
        self.doc_stats['average_generation_time_ms'] = ((current_avg * (total_docs - 1)) + generation_time) / total_docs
        
        self.logger.info(f"✅ Generated documentation: {doc_id} for {target_persona.value} ({generation_time:.1f}ms)")
        
        return documentation
    
    def export_documentation(self, documentation: PlanDocumentation,
                           output_format: DocumentFormat) -> str:
        """
        Export documentation to specified format
        
        Args:
            documentation: Plan documentation to export
            output_format: Target output format
            
        Returns:
            Exported documentation as string
        """
        if output_format == DocumentFormat.MARKDOWN:
            return self._export_to_markdown(documentation)
        elif output_format == DocumentFormat.HTML:
            return self._export_to_html(documentation)
        elif output_format == DocumentFormat.JSON:
            return json.dumps(asdict(documentation), indent=2, default=str)
        elif output_format == DocumentFormat.PDF:
            # PDF generation would be handled by external service
            return self._export_to_pdf_placeholder(documentation)
        else:
            return self._export_to_markdown(documentation)  # Default to Markdown
    
    def generate_multi_persona_documentation(self, plan_manifest: Dict[str, Any]) -> Dict[PersonaType, PlanDocumentation]:
        """
        Generate documentation for multiple personas
        
        Args:
            plan_manifest: Plan manifest to document
            
        Returns:
            Dictionary mapping personas to their documentation
        """
        docs = {}
        
        personas = [PersonaType.DEVELOPER, PersonaType.CRO, PersonaType.COMPLIANCE]
        
        for persona in personas:
            docs[persona] = self.generate_plan_documentation(plan_manifest, persona)
        
        return docs
    
    def get_documentation_statistics(self) -> Dict[str, Any]:
        """Get documentation service statistics"""
        return {
            **self.doc_stats,
            'cached_documents': len(self.document_cache),
            'supported_personas': len(PersonaType),
            'supported_formats': len(DocumentFormat)
        }
    
    def _generate_executive_summary(self, plan_manifest: Dict[str, Any]) -> ExecutiveSummary:
        """Generate executive summary from plan manifest"""
        # Extract basic information
        plan_graph = plan_manifest.get('plan_graph', {})
        nodes = plan_graph.get('nodes', [])
        metadata = plan_manifest.get('metadata', {})
        provenance = plan_manifest.get('provenance', {})
        
        # Count different node types
        total_nodes = len(nodes)
        ml_nodes = len([n for n in nodes if 'ml_' in n.get('type', '').lower() or n.get('type') == 'ml_node'])
        external_integrations = len([n for n in nodes if 'external' in n.get('type', '').lower() or 'api' in n.get('type', '').lower()])
        
        # Determine risk tier based on complexity and ML usage
        if ml_nodes > 5 or total_nodes > 100:
            risk_tier = RiskTier.HIGH
        elif ml_nodes > 2 or total_nodes > 50:
            risk_tier = RiskTier.MEDIUM
        elif ml_nodes > 0 or total_nodes > 20:
            risk_tier = RiskTier.LOW
        else:
            risk_tier = RiskTier.LOW
        
        # Extract compliance requirements
        compliance_requirements = []
        policies = plan_manifest.get('policies', {})
        if 'gdpr_compliance' in policies:
            compliance_requirements.append('GDPR')
        if 'sox_compliance' in policies:
            compliance_requirements.append('SOX')
        if 'pci_compliance' in policies:
            compliance_requirements.append('PCI-DSS')
        
        # Extract data residency
        data_residency_regions = []
        for node in nodes:
            node_policies = node.get('policies', {})
            if 'residency' in node_policies:
                region = node_policies['residency']
                if region not in data_residency_regions:
                    data_residency_regions.append(region)
        
        # Estimate cost (simplified)
        estimated_cost = self._estimate_monthly_cost(plan_manifest)
        
        return ExecutiveSummary(
            plan_purpose=metadata.get('purpose', 'Automated business process execution'),
            business_impact=metadata.get('business_impact', 'Streamlines operations and improves efficiency'),
            risk_tier=risk_tier,
            total_nodes=total_nodes,
            ml_nodes=ml_nodes,
            external_integrations=external_integrations,
            estimated_cost_monthly=estimated_cost,
            compliance_requirements=compliance_requirements,
            data_residency_regions=data_residency_regions,
            author=provenance.get('author'),
            approver=provenance.get('approver'),
            created_date=provenance.get('created_at'),
            last_modified=provenance.get('updated_at')
        )
    
    def _generate_node_documentation(self, plan_manifest: Dict[str, Any],
                                   target_persona: PersonaType,
                                   detail_level: str) -> List[NodeDocumentation]:
        """Generate documentation for all nodes"""
        plan_graph = plan_manifest.get('plan_graph', {})
        nodes = plan_graph.get('nodes', [])
        
        node_docs = []
        
        for node in nodes:
            node_doc = NodeDocumentation(
                node_id=node.get('id', 'unknown'),
                node_type=node.get('type', 'unknown'),
                display_name=node.get('name', node.get('id', 'Unnamed Node')),
                description=node.get('description', 'No description provided')
            )
            
            # Extract inputs and outputs
            node_doc.inputs = [
                {
                    'name': inp.get('name', 'unnamed'),
                    'type': inp.get('type', 'unknown'),
                    'description': inp.get('description', '')
                }
                for inp in node.get('inputs', [])
            ]
            
            node_doc.outputs = [
                {
                    'name': out.get('name', 'unnamed'),
                    'type': out.get('type', 'unknown'),
                    'description': out.get('description', '')
                }
                for out in node.get('outputs', [])
            ]
            
            # Business context (persona-dependent)
            if target_persona in [PersonaType.CRO, PersonaType.STAKEHOLDER]:
                node_doc.business_purpose = self._generate_business_purpose(node)
                node_doc.data_processing_description = self._generate_data_processing_description(node)
            
            # Extract policies
            node_doc.policies = node.get('policies', {})
            
            # Compliance notes
            node_doc.compliance_notes = self._generate_compliance_notes(node, target_persona)
            
            # SLA and cost information
            node_doc.sla_requirements = node.get('sla', {})
            node_doc.cost_class = node.get('cost_class', 'T2')  # Default to T2
            
            # ML-specific information
            if 'ml_' in node.get('type', '').lower() or node.get('type') == 'ml_node':
                node_doc.model_info = node.get('model_config', {})
                node_doc.trust_budget = node.get('trust_budget', {})
                node_doc.explainability_config = node.get('explainability', {})
                
                # Fallback strategy
                fallbacks = node.get('fallbacks', [])
                if fallbacks:
                    fallback_descriptions = []
                    for fallback in fallbacks:
                        condition = fallback.get('condition', 'default')
                        target = fallback.get('target', 'unknown')
                        fallback_descriptions.append(f"{condition} → {target}")
                    node_doc.fallback_strategy = "; ".join(fallback_descriptions)
            
            node_docs.append(node_doc)
        
        # Sort nodes by importance for the persona
        if target_persona == PersonaType.CRO:
            # CRO cares about business impact and cost
            node_docs.sort(key=lambda x: (x.cost_class, -len(x.business_purpose)))
        elif target_persona == PersonaType.COMPLIANCE:
            # Compliance cares about risk and policies
            node_docs.sort(key=lambda x: (-len(x.policies), -len(x.compliance_notes)))
        else:
            # Default: sort by node type and ID
            node_docs.sort(key=lambda x: (x.node_type, x.node_id))
        
        return node_docs
    
    def _generate_policy_impact_summary(self, plan_manifest: Dict[str, Any]) -> PolicyImpactSummary:
        """Generate policy impact summary"""
        summary = PolicyImpactSummary()
        
        # Extract global policies
        global_policies = plan_manifest.get('policies', {})
        
        # Extract node-level policies
        plan_graph = plan_manifest.get('plan_graph', {})
        nodes = plan_graph.get('nodes', [])
        
        all_policies = {}
        all_policies.update(global_policies)
        
        for node in nodes:
            node_policies = node.get('policies', {})
            for key, value in node_policies.items():
                if key not in all_policies:
                    all_policies[key] = []
                if isinstance(all_policies[key], list):
                    if value not in all_policies[key]:
                        all_policies[key].append(value)
                else:
                    all_policies[key] = [all_policies[key], value]
        
        # Analyze residency constraints
        if 'residency' in all_policies:
            residency_values = all_policies['residency']
            if isinstance(residency_values, list):
                summary.data_residency_constraints = residency_values
            else:
                summary.data_residency_constraints = [residency_values]
        
        # Analyze compliance requirements
        compliance_keys = ['gdpr_compliance', 'sox_compliance', 'pci_compliance', 'hipaa_compliance']
        for key in compliance_keys:
            if key in all_policies and all_policies[key]:
                compliance_name = key.replace('_compliance', '').upper()
                summary.regulatory_compliance.append(compliance_name)
        
        # Analyze access controls
        if 'access_control' in all_policies:
            summary.access_controls.append(str(all_policies['access_control']))
        
        # Analyze audit requirements
        if 'audit_level' in all_policies:
            summary.audit_requirements.append(f"Audit level: {all_policies['audit_level']}")
        
        # Calculate fallback coverage
        nodes_with_fallbacks = len([n for n in nodes if n.get('fallbacks')])
        ml_nodes = len([n for n in nodes if 'ml_' in n.get('type', '').lower() or n.get('type') == 'ml_node'])
        
        if ml_nodes > 0:
            summary.fallback_coverage = (nodes_with_fallbacks / ml_nodes) * 100
        
        # ML governance policies
        ml_policy_keys = ['model_governance', 'explainability', 'bias_monitoring', 'drift_detection']
        for key in ml_policy_keys:
            if key in all_policies:
                summary.ml_governance_policies.append(f"{key}: {all_policies[key]}")
        
        return summary
    
    def _generate_cost_summary(self, plan_manifest: Dict[str, Any]) -> CostSummary:
        """Generate cost analysis summary"""
        summary = CostSummary()
        
        plan_graph = plan_manifest.get('plan_graph', {})
        nodes = plan_graph.get('nodes', [])
        
        # Estimate costs by node type
        cost_by_type = defaultdict(float)
        
        for node in nodes:
            node_type = node.get('type', 'unknown')
            cost_class = node.get('cost_class', 'T2')
            
            # Use cost model to estimate
            base_cost = self.cost_models.get(cost_class, 0.10)  # Default $0.10/hour
            
            # ML nodes are more expensive
            if 'ml_' in node_type.lower() or node_type == 'ml_node':
                base_cost *= 10  # 10x multiplier for ML
                summary.ml_inference_costs += base_cost * 24 * 30  # Monthly cost
            
            # External API calls have additional costs
            if 'external' in node_type.lower() or 'api' in node_type.lower():
                base_cost += 0.001  # $0.001 per call
            
            monthly_cost = base_cost * 24 * 30  # Assume 24/7 operation
            cost_by_type[node_type] += monthly_cost
        
        summary.cost_by_node_type = dict(cost_by_type)
        summary.estimated_monthly_cost = sum(cost_by_type.values())
        
        # Estimate resource usage
        summary.estimated_compute_hours = len(nodes) * 24 * 30  # Node-hours per month
        summary.estimated_storage_gb = len(nodes) * 0.1  # 100MB per node
        summary.estimated_network_gb = len(nodes) * 1.0  # 1GB per node per month
        
        # Cost categories
        summary.cost_by_category = {
            'compute': summary.estimated_monthly_cost * 0.6,
            'storage': summary.estimated_storage_gb * 0.023,  # $0.023/GB/month
            'network': summary.estimated_network_gb * 0.09,   # $0.09/GB
            'ml_inference': summary.ml_inference_costs
        }
        
        # Optimization suggestions
        if summary.estimated_monthly_cost > 1000:
            summary.optimization_opportunities.append("Consider reserved capacity for cost savings")
        
        if summary.ml_inference_costs > summary.estimated_monthly_cost * 0.5:
            summary.optimization_opportunities.append("ML costs are high - consider model optimization")
        
        if len(nodes) > 50:
            summary.optimization_opportunities.append("Large plan - consider modularization")
        
        return summary
    
    def _generate_sla_summary(self, plan_manifest: Dict[str, Any]) -> SLASummary:
        """Generate SLA requirements summary"""
        summary = SLASummary()
        
        metadata = plan_manifest.get('metadata', {})
        
        # Extract SLA requirements from metadata
        summary.target_availability = metadata.get('target_availability', '99.9%')
        summary.target_response_time = metadata.get('target_response_time', '< 100ms')
        summary.target_throughput = metadata.get('target_throughput', '1000 req/min')
        
        # Recovery objectives
        summary.rto_target = metadata.get('rto_target', '< 5 minutes')
        summary.rpo_target = metadata.get('rpo_target', '< 1 minute')
        
        # Monitoring requirements
        plan_graph = plan_manifest.get('plan_graph', {})
        nodes = plan_graph.get('nodes', [])
        
        ml_nodes = [n for n in nodes if 'ml_' in n.get('type', '').lower() or n.get('type') == 'ml_node']
        external_nodes = [n for n in nodes if 'external' in n.get('type', '').lower()]
        
        if ml_nodes:
            summary.monitoring_requirements.append("ML model performance monitoring")
            summary.monitoring_requirements.append("Confidence threshold monitoring")
            summary.alert_thresholds['ml_confidence'] = '< 0.8'
        
        if external_nodes:
            summary.monitoring_requirements.append("External API availability monitoring")
            summary.alert_thresholds['api_response_time'] = '> 5000ms'
        
        # Dependencies
        for node in external_nodes:
            node_id = node.get('id', 'unknown')
            summary.critical_dependencies.append(node_id)
        
        return summary
    
    def _generate_evidence_summary(self, plan_manifest: Dict[str, Any]) -> EvidenceCheckpointSummary:
        """Generate evidence checkpoint summary"""
        summary = EvidenceCheckpointSummary()
        
        metadata = plan_manifest.get('metadata', {})
        policies = plan_manifest.get('policies', {})
        
        # Evidence collection configuration
        if 'evidence_config' in metadata:
            evidence_config = metadata['evidence_config']
            summary.checkpoint_frequency = evidence_config.get('frequency', 'per_node')
            summary.data_retention_period = evidence_config.get('retention', '7 years')
        
        # Evidence types
        summary.evidence_types_collected = [
            'execution_logs',
            'input_snapshots',
            'output_snapshots',
            'decision_metadata',
            'performance_metrics'
        ]
        
        # Check for ML nodes - they need additional evidence
        plan_graph = plan_manifest.get('plan_graph', {})
        nodes = plan_graph.get('nodes', [])
        ml_nodes = [n for n in nodes if 'ml_' in n.get('type', '').lower() or n.get('type') == 'ml_node']
        
        if ml_nodes:
            summary.evidence_types_collected.extend([
                'model_predictions',
                'confidence_scores',
                'explainability_outputs',
                'bias_metrics'
            ])
        
        # Regulatory evidence mapping
        compliance_requirements = []
        if 'gdpr_compliance' in policies:
            compliance_requirements.append('GDPR')
        if 'sox_compliance' in policies:
            compliance_requirements.append('SOX')
        
        for requirement in compliance_requirements:
            summary.regulatory_evidence_mapping[requirement] = [
                'audit_trail',
                'access_logs',
                'data_processing_records'
            ]
        
        return summary
    
    def _generate_business_purpose(self, node: Dict[str, Any]) -> str:
        """Generate business purpose description for a node"""
        node_type = node.get('type', 'unknown')
        node_name = node.get('name', node.get('id', 'Unknown'))
        
        # Business purpose templates by node type
        purpose_templates = {
            'ml_node': f"Provides intelligent decision-making capabilities for {node_name}",
            'decision': f"Implements business logic and routing decisions for {node_name}",
            'external_api': f"Integrates with external systems to retrieve or update data for {node_name}",
            'data_transform': f"Processes and transforms data to meet business requirements for {node_name}",
            'validation': f"Ensures data quality and compliance requirements for {node_name}",
            'notification': f"Communicates results and status updates for {node_name}"
        }
        
        return purpose_templates.get(node_type, f"Performs {node_type} operations for {node_name}")
    
    def _generate_data_processing_description(self, node: Dict[str, Any]) -> str:
        """Generate data processing description for a node"""
        inputs = node.get('inputs', [])
        outputs = node.get('outputs', [])
        
        if not inputs and not outputs:
            return "No data processing specified"
        
        input_desc = f"Processes {len(inputs)} input(s)" if inputs else "No inputs"
        output_desc = f"produces {len(outputs)} output(s)" if outputs else "no outputs"
        
        return f"{input_desc} and {output_desc}"
    
    def _generate_compliance_notes(self, node: Dict[str, Any], 
                                 target_persona: PersonaType) -> List[str]:
        """Generate compliance notes for a node"""
        notes = []
        
        policies = node.get('policies', {})
        node_type = node.get('type', 'unknown')
        
        # Check for data residency requirements
        if 'residency' in policies:
            notes.append(f"Data must remain in {policies['residency']} region")
        
        # Check for ML-specific compliance
        if 'ml_' in node_type.lower() or node_type == 'ml_node':
            if target_persona == PersonaType.COMPLIANCE:
                notes.append("ML model requires bias monitoring and explainability")
                notes.append("Model predictions must be auditable")
            
            if 'explainability' in policies:
                notes.append(f"Explainability level: {policies['explainability']}")
        
        # Check for access control requirements
        if 'access_control' in policies:
            notes.append(f"Access control: {policies['access_control']}")
        
        return notes
    
    def _estimate_monthly_cost(self, plan_manifest: Dict[str, Any]) -> float:
        """Estimate monthly cost for the plan"""
        plan_graph = plan_manifest.get('plan_graph', {})
        nodes = plan_graph.get('nodes', [])
        
        total_cost = 0.0
        
        for node in nodes:
            node_type = node.get('type', 'unknown')
            cost_class = node.get('cost_class', 'T2')
            
            base_cost = self.cost_models.get(cost_class, 0.10)
            
            # ML nodes are more expensive
            if 'ml_' in node_type.lower() or node_type == 'ml_node':
                base_cost *= 10
            
            # Monthly cost (assume 24/7 operation)
            monthly_cost = base_cost * 24 * 30
            total_cost += monthly_cost
        
        return total_cost
    
    def _generate_topology_diagram_placeholder(self, plan_manifest: Dict[str, Any]) -> str:
        """Generate topology diagram placeholder"""
        # This would integrate with Task 6.2.44 (Graph Visualizer)
        plan_id = plan_manifest.get('plan_id', 'unknown')
        return f"[Topology Diagram for {plan_id} - Generated by Graph Visualizer Service]"
    
    def _generate_fallback_diagram_placeholder(self, plan_manifest: Dict[str, Any]) -> str:
        """Generate fallback diagram placeholder"""
        # This would integrate with Task 6.2.44 (Graph Visualizer)
        plan_id = plan_manifest.get('plan_id', 'unknown')
        return f"[Fallback DAG Diagram for {plan_id} - Generated by Graph Visualizer Service]"
    
    def _generate_data_flow_diagram_placeholder(self, plan_manifest: Dict[str, Any]) -> str:
        """Generate data flow diagram placeholder"""
        # This would integrate with Task 6.2.44 (Graph Visualizer)
        plan_id = plan_manifest.get('plan_id', 'unknown')
        return f"[Data Flow Diagram for {plan_id} - Generated by Graph Visualizer Service]"
    
    def _export_to_markdown(self, documentation: PlanDocumentation) -> str:
        """Export documentation to Markdown format"""
        lines = []
        
        # Title
        lines.append(f"# Plan Documentation: {documentation.plan_id}")
        lines.append(f"**Version:** {documentation.plan_version}")
        lines.append(f"**Generated:** {documentation.generated_at.strftime('%Y-%m-%d %H:%M:%S UTC')}")
        lines.append(f"**Target Audience:** {documentation.target_persona.value.title()}")
        lines.append("")
        
        # Executive Summary
        lines.append("## Executive Summary")
        summary = documentation.executive_summary
        lines.append(f"**Purpose:** {summary.plan_purpose}")
        lines.append(f"**Business Impact:** {summary.business_impact}")
        lines.append(f"**Risk Tier:** {summary.risk_tier.value.title()}")
        lines.append("")
        lines.append("### Key Metrics")
        lines.append(f"- **Total Nodes:** {summary.total_nodes}")
        lines.append(f"- **ML Nodes:** {summary.ml_nodes}")
        lines.append(f"- **External Integrations:** {summary.external_integrations}")
        lines.append(f"- **Estimated Monthly Cost:** ${summary.estimated_cost_monthly:.2f}")
        lines.append("")
        
        if summary.compliance_requirements:
            lines.append("### Compliance Requirements")
            for req in summary.compliance_requirements:
                lines.append(f"- {req}")
            lines.append("")
        
        # Node Documentation
        lines.append("## Node Documentation")
        for node_doc in documentation.node_documentation:
            lines.append(f"### {node_doc.display_name} ({node_doc.node_id})")
            lines.append(f"**Type:** {node_doc.node_type}")
            lines.append(f"**Description:** {node_doc.description}")
            
            if node_doc.business_purpose:
                lines.append(f"**Business Purpose:** {node_doc.business_purpose}")
            
            if node_doc.inputs:
                lines.append("**Inputs:**")
                for inp in node_doc.inputs:
                    lines.append(f"- {inp['name']} ({inp['type']})")
            
            if node_doc.outputs:
                lines.append("**Outputs:**")
                for out in node_doc.outputs:
                    lines.append(f"- {out['name']} ({out['type']})")
            
            if node_doc.policies:
                lines.append("**Policies:**")
                for key, value in node_doc.policies.items():
                    lines.append(f"- {key}: {value}")
            
            if node_doc.fallback_strategy:
                lines.append(f"**Fallback Strategy:** {node_doc.fallback_strategy}")
            
            lines.append("")
        
        # Policy Impact Summary
        lines.append("## Policy Impact Summary")
        policy_summary = documentation.policy_impact_summary
        
        if policy_summary.data_residency_constraints:
            lines.append("### Data Residency")
            for constraint in policy_summary.data_residency_constraints:
                lines.append(f"- {constraint}")
        
        if policy_summary.regulatory_compliance:
            lines.append("### Regulatory Compliance")
            for compliance in policy_summary.regulatory_compliance:
                lines.append(f"- {compliance}")
        
        if policy_summary.ml_governance_policies:
            lines.append("### ML Governance")
            for policy in policy_summary.ml_governance_policies:
                lines.append(f"- {policy}")
        
        lines.append(f"**Fallback Coverage:** {policy_summary.fallback_coverage:.1f}%")
        lines.append("")
        
        # Cost Summary
        lines.append("## Cost Summary")
        cost_summary = documentation.cost_summary
        lines.append(f"**Estimated Monthly Cost:** ${cost_summary.estimated_monthly_cost:.2f}")
        
        if cost_summary.cost_by_node_type:
            lines.append("### Cost by Node Type")
            for node_type, cost in cost_summary.cost_by_node_type.items():
                lines.append(f"- {node_type}: ${cost:.2f}")
        
        if cost_summary.optimization_opportunities:
            lines.append("### Optimization Opportunities")
            for opportunity in cost_summary.optimization_opportunities:
                lines.append(f"- {opportunity}")
        
        lines.append("")
        
        # SLA Summary
        lines.append("## SLA Summary")
        sla_summary = documentation.sla_summary
        lines.append(f"**Target Availability:** {sla_summary.target_availability}")
        lines.append(f"**Target Response Time:** {sla_summary.target_response_time}")
        lines.append(f"**Target Throughput:** {sla_summary.target_throughput}")
        lines.append("")
        
        # Evidence Summary
        lines.append("## Evidence and Audit")
        evidence_summary = documentation.evidence_summary
        lines.append(f"**Checkpoint Frequency:** {evidence_summary.checkpoint_frequency}")
        lines.append(f"**Data Retention:** {evidence_summary.data_retention_period}")
        
        if evidence_summary.evidence_types_collected:
            lines.append("### Evidence Types Collected")
            for evidence_type in evidence_summary.evidence_types_collected:
                lines.append(f"- {evidence_type}")
        
        lines.append("")
        
        # Diagrams
        if documentation.topology_diagram:
            lines.append("## Topology Diagram")
            lines.append(documentation.topology_diagram)
            lines.append("")
        
        if documentation.fallback_diagram:
            lines.append("## Fallback Diagram")
            lines.append(documentation.fallback_diagram)
            lines.append("")
        
        return "\n".join(lines)
    
    def _export_to_html(self, documentation: PlanDocumentation) -> str:
        """Export documentation to HTML format"""
        # Convert Markdown to HTML (simplified)
        markdown_content = self._export_to_markdown(documentation)
        
        html_parts = [
            "<!DOCTYPE html>",
            "<html>",
            "<head>",
            f"<title>Plan Documentation: {documentation.plan_id}</title>",
            "<style>",
            "body { font-family: Arial, sans-serif; margin: 40px; }",
            "h1, h2, h3 { color: #333; }",
            "table { border-collapse: collapse; width: 100%; }",
            "th, td { border: 1px solid #ddd; padding: 8px; text-align: left; }",
            "th { background-color: #f2f2f2; }",
            ".summary { background-color: #f9f9f9; padding: 20px; border-radius: 5px; }",
            ".node { margin: 20px 0; padding: 15px; border: 1px solid #eee; }",
            "</style>",
            "</head>",
            "<body>"
        ]
        
        # Convert markdown to basic HTML
        html_content = markdown_content.replace('\n## ', '\n<h2>').replace('\n### ', '\n<h3>')
        html_content = html_content.replace('\n# ', '\n<h1>')
        html_content = html_content.replace('**', '<strong>').replace('**', '</strong>')
        html_content = html_content.replace('\n- ', '\n<li>').replace('\n\n', '</ul>\n<p>')
        html_content = html_content.replace('\n', '<br>\n')
        
        html_parts.append(html_content)
        html_parts.extend(["</body>", "</html>"])
        
        return "\n".join(html_parts)
    
    def _export_to_pdf_placeholder(self, documentation: PlanDocumentation) -> str:
        """Export documentation to PDF (placeholder)"""
        # PDF generation would be handled by external service
        return f"[PDF Export Placeholder for {documentation.plan_id} - Would be generated by PDF service]"
    
    def _initialize_persona_templates(self) -> Dict[PersonaType, Dict[str, Any]]:
        """Initialize persona-specific templates"""
        return {
            PersonaType.DEVELOPER: {
                'focus_areas': ['technical_details', 'implementation', 'debugging'],
                'detail_level': 'comprehensive',
                'include_diagrams': True,
                'include_code_snippets': True
            },
            PersonaType.CRO: {
                'focus_areas': ['business_impact', 'cost', 'roi'],
                'detail_level': 'standard',
                'include_diagrams': True,
                'include_business_metrics': True
            },
            PersonaType.COMPLIANCE: {
                'focus_areas': ['policies', 'audit', 'risk', 'governance'],
                'detail_level': 'comprehensive',
                'include_diagrams': False,
                'include_compliance_mapping': True
            },
            PersonaType.AUDITOR: {
                'focus_areas': ['evidence', 'controls', 'traceability'],
                'detail_level': 'comprehensive',
                'include_diagrams': False,
                'include_audit_trail': True
            }
        }
    
    def _initialize_cost_models(self) -> Dict[str, float]:
        """Initialize cost estimation models"""
        return {
            'T0': 0.05,   # $0.05/hour - very light
            'T1': 0.10,   # $0.10/hour - light
            'T2': 0.20,   # $0.20/hour - standard
            'T3': 0.50,   # $0.50/hour - heavy
            'T4': 1.00    # $1.00/hour - very heavy
        }
    
    def _initialize_sla_templates(self) -> Dict[str, Dict[str, str]]:
        """Initialize SLA templates"""
        return {
            'high_availability': {
                'availability': '99.99%',
                'response_time': '< 50ms',
                'throughput': '10000 req/min'
            },
            'standard': {
                'availability': '99.9%',
                'response_time': '< 100ms',
                'throughput': '1000 req/min'
            },
            'basic': {
                'availability': '99%',
                'response_time': '< 500ms',
                'throughput': '100 req/min'
            }
        }

# API Interface
class DocumentationGeneratorAPI:
    """API interface for documentation generation operations"""
    
    def __init__(self, doc_service: Optional[DocumentationGeneratorService] = None):
        self.doc_service = doc_service or DocumentationGeneratorService()
    
    def generate_documentation(self, plan_manifest: Dict[str, Any],
                             persona: str = "stakeholder",
                             detail_level: str = "standard") -> Dict[str, Any]:
        """API endpoint to generate plan documentation"""
        try:
            persona_enum = PersonaType(persona)
            documentation = self.doc_service.generate_plan_documentation(
                plan_manifest, persona_enum, detail_level
            )
            
            return {
                'success': True,
                'documentation': asdict(documentation)
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': str(e),
                'error_type': type(e).__name__
            }
    
    def export_documentation(self, documentation_dict: Dict[str, Any],
                           output_format: str = "markdown") -> Dict[str, Any]:
        """API endpoint to export documentation"""
        try:
            documentation = PlanDocumentation(**documentation_dict)
            format_enum = DocumentFormat(output_format)
            
            exported_content = self.doc_service.export_documentation(documentation, format_enum)
            
            return {
                'success': True,
                'content': exported_content,
                'format': output_format
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': str(e)
            }
    
    def generate_multi_persona_docs(self, plan_manifest: Dict[str, Any]) -> Dict[str, Any]:
        """API endpoint to generate documentation for multiple personas"""
        try:
            docs = self.doc_service.generate_multi_persona_documentation(plan_manifest)
            
            return {
                'success': True,
                'documentation': {
                    persona.value: asdict(doc) for persona, doc in docs.items()
                }
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': str(e)
            }

# Test Functions
def create_test_plan_manifest() -> Dict[str, Any]:
    """Create a test plan manifest for documentation generation"""
    return {
        'plan_id': 'customer_onboarding_automation',
        'version': '2.1.0',
        'metadata': {
            'purpose': 'Automate customer onboarding process with ML-driven risk assessment',
            'business_impact': 'Reduces onboarding time by 70% and improves risk detection accuracy',
            'target_availability': '99.95%',
            'target_response_time': '< 200ms',
            'environment': 'production',
            'cost_estimate': '$2,450/month'
        },
        'provenance': {
            'author': 'john.doe@company.com',
            'approver': 'jane.smith@company.com',
            'created_at': '2024-01-15T10:30:00Z',
            'updated_at': '2024-01-20T14:45:00Z'
        },
        'plan_graph': {
            'nodes': [
                {
                    'id': 'customer_data_ingestion',
                    'type': 'data_ingestion',
                    'name': 'Customer Data Ingestion',
                    'description': 'Ingests customer application data from multiple sources',
                    'inputs': [
                        {'name': 'application_form', 'type': 'json', 'description': 'Customer application form data'},
                        {'name': 'identity_documents', 'type': 'file_array', 'description': 'Identity verification documents'}
                    ],
                    'outputs': [
                        {'name': 'structured_data', 'type': 'json', 'description': 'Normalized customer data'},
                        {'name': 'document_metadata', 'type': 'json', 'description': 'Document processing metadata'}
                    ],
                    'cost_class': 'T1',
                    'policies': {
                        'residency': 'us-east-1',
                        'pii_handling': 'encrypted',
                        'audit_level': 'full'
                    }
                },
                {
                    'id': 'risk_assessment_ml',
                    'type': 'ml_node',
                    'name': 'ML Risk Assessment',
                    'description': 'ML-powered risk assessment of customer applications',
                    'inputs': [
                        {'name': 'customer_features', 'type': 'vector', 'description': 'Engineered customer features'},
                        {'name': 'historical_data', 'type': 'json', 'description': 'Historical customer data'}
                    ],
                    'outputs': [
                        {'name': 'risk_score', 'type': 'float', 'description': 'Risk score (0-1)'},
                        {'name': 'risk_category', 'type': 'enum', 'description': 'Risk category (low/medium/high)'},
                        {'name': 'confidence', 'type': 'float', 'description': 'Model confidence score'}
                    ],
                    'cost_class': 'T3',
                    'model_config': {
                        'model_id': 'risk_classifier_v2.1',
                        'model_type': 'gradient_boosting',
                        'version': '2.1.0'
                    },
                    'trust_budget': {
                        'min_confidence': 0.8,
                        'auto_execute_threshold': 0.9,
                        'manual_review_threshold': 0.7
                    },
                    'fallbacks': [
                        {
                            'condition': 'confidence < 0.8',
                            'target': 'manual_risk_review',
                            'reason': 'Low confidence prediction'
                        },
                        {
                            'condition': 'risk_score > 0.9',
                            'target': 'enhanced_verification',
                            'reason': 'High risk customer'
                        }
                    ],
                    'explainability': {
                        'method': 'shap',
                        'enabled': True,
                        'audit_integration': True
                    },
                    'policies': {
                        'model_governance': 'strict',
                        'bias_monitoring': 'enabled',
                        'drift_detection': 'enabled',
                        'explainability': 'required'
                    }
                },
                {
                    'id': 'decision_engine',
                    'type': 'decision',
                    'name': 'Onboarding Decision Engine',
                    'description': 'Makes final onboarding decision based on risk assessment',
                    'inputs': [
                        {'name': 'risk_score', 'type': 'float'},
                        {'name': 'risk_category', 'type': 'enum'},
                        {'name': 'confidence', 'type': 'float'}
                    ],
                    'outputs': [
                        {'name': 'decision', 'type': 'enum', 'description': 'approve/reject/review'},
                        {'name': 'reason', 'type': 'string', 'description': 'Decision rationale'}
                    ],
                    'cost_class': 'T1'
                },
                {
                    'id': 'notification_service',
                    'type': 'external_api',
                    'name': 'Customer Notification',
                    'description': 'Notifies customer of onboarding decision',
                    'inputs': [
                        {'name': 'customer_email', 'type': 'string'},
                        {'name': 'decision', 'type': 'enum'},
                        {'name': 'reason', 'type': 'string'}
                    ],
                    'outputs': [
                        {'name': 'notification_sent', 'type': 'boolean'}
                    ],
                    'cost_class': 'T1',
                    'sla': {
                        'response_time': '< 5s',
                        'availability': '99.9%'
                    }
                }
            ],
            'edges': [
                {'source': 'customer_data_ingestion', 'target': 'risk_assessment_ml'},
                {'source': 'risk_assessment_ml', 'target': 'decision_engine'},
                {'source': 'decision_engine', 'target': 'notification_service'}
            ]
        },
        'policies': {
            'gdpr_compliance': True,
            'sox_compliance': True,
            'audit_level': 'full',
            'data_retention': '7_years',
            'error_handling': 'fail_safe'
        }
    }

def run_documentation_generator_tests():
    """Run comprehensive documentation generator tests"""
    print("=== Documentation Generator Service Tests ===")
    
    # Initialize service
    doc_service = DocumentationGeneratorService()
    doc_api = DocumentationGeneratorAPI(doc_service)
    
    # Create test plan manifest
    test_manifest = create_test_plan_manifest()
    
    # Test 1: Basic documentation generation
    print("\n1. Testing basic documentation generation...")
    
    personas = [PersonaType.DEVELOPER, PersonaType.CRO, PersonaType.COMPLIANCE]
    
    for persona in personas:
        documentation = doc_service.generate_plan_documentation(test_manifest, persona)
        
        print(f"   {persona.value} documentation:")
        print(f"     Plan ID: {documentation.plan_id}")
        print(f"     Risk tier: {documentation.executive_summary.risk_tier.value}")
        print(f"     Total nodes: {documentation.executive_summary.total_nodes}")
        print(f"     ML nodes: {documentation.executive_summary.ml_nodes}")
        print(f"     Estimated cost: ${documentation.cost_summary.estimated_monthly_cost:.2f}")
        print(f"     Node documentation: {len(documentation.node_documentation)} nodes")
    
    # Test 2: Executive summary analysis
    print("\n2. Testing executive summary analysis...")
    
    stakeholder_doc = doc_service.generate_plan_documentation(test_manifest, PersonaType.STAKEHOLDER)
    summary = stakeholder_doc.executive_summary
    
    print(f"   Executive summary:")
    print(f"     Purpose: {summary.plan_purpose}")
    print(f"     Business impact: {summary.business_impact}")
    print(f"     Risk tier: {summary.risk_tier.value}")
    print(f"     Compliance requirements: {summary.compliance_requirements}")
    print(f"     Data residency regions: {summary.data_residency_regions}")
    print(f"     Author: {summary.author}")
    
    # Test 3: Node documentation details
    print("\n3. Testing node documentation details...")
    
    developer_doc = doc_service.generate_plan_documentation(test_manifest, PersonaType.DEVELOPER)
    
    for node_doc in developer_doc.node_documentation[:2]:  # Show first 2 nodes
        print(f"   Node: {node_doc.display_name}")
        print(f"     Type: {node_doc.node_type}")
        print(f"     Business purpose: {node_doc.business_purpose}")
        print(f"     Inputs: {len(node_doc.inputs)}")
        print(f"     Outputs: {len(node_doc.outputs)}")
        print(f"     Policies: {len(node_doc.policies)}")
        if node_doc.fallback_strategy:
            print(f"     Fallback strategy: {node_doc.fallback_strategy}")
    
    # Test 4: Policy impact summary
    print("\n4. Testing policy impact summary...")
    
    compliance_doc = doc_service.generate_plan_documentation(test_manifest, PersonaType.COMPLIANCE)
    policy_summary = compliance_doc.policy_impact_summary
    
    print(f"   Policy impact summary:")
    print(f"     Regulatory compliance: {policy_summary.regulatory_compliance}")
    print(f"     Data residency constraints: {policy_summary.data_residency_constraints}")
    print(f"     ML governance policies: {len(policy_summary.ml_governance_policies)}")
    print(f"     Fallback coverage: {policy_summary.fallback_coverage:.1f}%")
    
    # Test 5: Cost analysis
    print("\n5. Testing cost analysis...")
    
    cro_doc = doc_service.generate_plan_documentation(test_manifest, PersonaType.CRO)
    cost_summary = cro_doc.cost_summary
    
    print(f"   Cost summary:")
    print(f"     Estimated monthly cost: ${cost_summary.estimated_monthly_cost:.2f}")
    print(f"     ML inference costs: ${cost_summary.ml_inference_costs:.2f}")
    print(f"     Cost by node type: {cost_summary.cost_by_node_type}")
    print(f"     Optimization opportunities: {len(cost_summary.optimization_opportunities)}")
    
    # Test 6: Export to different formats
    print("\n6. Testing export to different formats...")
    
    base_doc = doc_service.generate_plan_documentation(test_manifest)
    
    formats = [DocumentFormat.MARKDOWN, DocumentFormat.HTML, DocumentFormat.JSON]
    
    for output_format in formats:
        exported_content = doc_service.export_documentation(base_doc, output_format)
        
        print(f"   {output_format.value} export:")
        print(f"     Content length: {len(exported_content)} characters")
        print(f"     Preview: {exported_content[:100].replace(chr(10), ' ')}...")
    
    # Test 7: Multi-persona documentation
    print("\n7. Testing multi-persona documentation...")
    
    multi_docs = doc_service.generate_multi_persona_documentation(test_manifest)
    
    print(f"   Multi-persona documentation:")
    for persona, doc in multi_docs.items():
        print(f"     {persona.value}: {len(doc.node_documentation)} nodes, ${doc.cost_summary.estimated_monthly_cost:.2f}")
    
    # Test 8: SLA and evidence summaries
    print("\n8. Testing SLA and evidence summaries...")
    
    sla_summary = base_doc.sla_summary
    evidence_summary = base_doc.evidence_summary
    
    print(f"   SLA summary:")
    print(f"     Target availability: {sla_summary.target_availability}")
    print(f"     Target response time: {sla_summary.target_response_time}")
    print(f"     Monitoring requirements: {len(sla_summary.monitoring_requirements)}")
    
    print(f"   Evidence summary:")
    print(f"     Checkpoint frequency: {evidence_summary.checkpoint_frequency}")
    print(f"     Evidence types: {len(evidence_summary.evidence_types_collected)}")
    print(f"     Regulatory mapping: {len(evidence_summary.regulatory_evidence_mapping)}")
    
    # Test 9: API interface
    print("\n9. Testing API interface...")
    
    # Test API documentation generation
    api_doc_result = doc_api.generate_documentation(test_manifest, "cro", "detailed")
    print(f"   API documentation: {'✅ PASS' if api_doc_result['success'] else '❌ FAIL'}")
    
    # Test API export
    if api_doc_result['success']:
        api_export_result = doc_api.export_documentation(
            api_doc_result['documentation'], "markdown"
        )
        print(f"   API export: {'✅ PASS' if api_export_result['success'] else '❌ FAIL'}")
    
    # Test API multi-persona
    api_multi_result = doc_api.generate_multi_persona_docs(test_manifest)
    print(f"   API multi-persona: {'✅ PASS' if api_multi_result['success'] else '❌ FAIL'}")
    if api_multi_result['success']:
        print(f"     Personas generated: {len(api_multi_result['documentation'])}")
    
    # Test 10: Edge cases and performance
    print("\n10. Testing edge cases and performance...")
    
    # Empty plan
    empty_manifest = {
        'plan_id': 'empty_plan',
        'version': '1.0.0',
        'plan_graph': {'nodes': [], 'edges': []}
    }
    
    empty_doc = doc_service.generate_plan_documentation(empty_manifest)
    print(f"   Empty plan documentation: {empty_doc.executive_summary.total_nodes} nodes")
    
    # Plan with only ML nodes
    ml_only_manifest = {
        'plan_id': 'ml_only_plan',
        'version': '1.0.0',
        'plan_graph': {
            'nodes': [
                {'id': 'ml1', 'type': 'ml_node', 'name': 'ML Node 1'},
                {'id': 'ml2', 'type': 'ml_node', 'name': 'ML Node 2'}
            ],
            'edges': [{'source': 'ml1', 'target': 'ml2'}]
        }
    }
    
    ml_doc = doc_service.generate_plan_documentation(ml_only_manifest)
    print(f"   ML-only plan: {ml_doc.executive_summary.ml_nodes} ML nodes, risk tier: {ml_doc.executive_summary.risk_tier.value}")
    
    # Test caching
    start_time = datetime.now(timezone.utc)
    for i in range(3):
        cached_doc = doc_service.generate_plan_documentation(test_manifest)
    end_time = datetime.now(timezone.utc)
    
    total_time = (end_time - start_time).total_seconds() * 1000
    print(f"   3 cached generations: {total_time:.1f}ms total")
    
    # Test statistics
    stats = doc_service.get_documentation_statistics()
    print(f"   Total documents generated: {stats['total_documents_generated']}")
    print(f"   Documents by persona: {stats['documents_by_persona']}")
    print(f"   Average generation time: {stats['average_generation_time_ms']:.1f}ms")
    print(f"   Cache hits: {stats['cache_hits']}")
    
    print(f"\n=== Test Summary ===")
    final_stats = doc_service.get_documentation_statistics()
    print(f"Documentation generator service tested successfully")
    print(f"Total documents: {final_stats['total_documents_generated']}")
    print(f"Personas supported: {final_stats['supported_personas']}")
    print(f"Formats supported: {final_stats['supported_formats']}")
    print(f"Cache hit rate: {final_stats['cache_hits']}/{final_stats['cache_hits'] + final_stats['cache_misses']}")
    print(f"Average generation time: {final_stats['average_generation_time_ms']:.1f}ms")
    
    return doc_service, doc_api

if __name__ == "__main__":
    run_documentation_generator_tests()

