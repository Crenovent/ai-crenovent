"""
IR Pretty-Printer Service - Task 6.2.42
========================================

IR pretty-printer for reviews
- Produces human-readable, well-formatted representation of IR
- Suitable for code review, PR diffs, and compliance reviews
- Preserves comments and shows provenance & policy annotations
- Supports diff mode to highlight changes between IRs
- Backend implementation (no actual PR system integration - that's infrastructure)

Dependencies: Task 6.2.4 (IR)
Outputs: Human-readable IR → enables effective code reviews and compliance audits
"""

from typing import Dict, List, Any, Optional, Union, Tuple, Set
import json
import logging
from datetime import datetime, timezone
from dataclasses import dataclass, field, asdict
from enum import Enum
import re
import difflib
from collections import defaultdict

logger = logging.getLogger(__name__)

class OutputFormat(Enum):
    """Output formats for pretty-printing"""
    TEXT = "text"
    MARKDOWN = "markdown"
    HTML = "html"
    JSON = "json"

class DiffMode(Enum):
    """Diff modes for comparing IRs"""
    UNIFIED = "unified"
    CONTEXT = "context"
    SIDE_BY_SIDE = "side_by_side"
    INLINE = "inline"

class AnnotationLevel(Enum):
    """Levels of annotation detail"""
    MINIMAL = "minimal"      # Basic structure only
    STANDARD = "standard"    # Include policies and metadata
    VERBOSE = "verbose"      # Include all available information
    DEBUG = "debug"          # Include internal details

@dataclass
class PrettyPrintOptions:
    """Options for pretty-printing IR"""
    # Output format
    output_format: OutputFormat = OutputFormat.TEXT
    
    # Content options
    annotation_level: AnnotationLevel = AnnotationLevel.STANDARD
    include_comments: bool = True
    include_provenance: bool = True
    include_policies: bool = True
    include_metadata: bool = True
    
    # Formatting options
    indent_size: int = 2
    max_line_length: int = 120
    compact_arrays: bool = False
    sort_keys: bool = True
    
    # Filtering options
    node_types_filter: Optional[List[str]] = None
    show_only_changed: bool = False
    highlight_keywords: List[str] = field(default_factory=list)
    
    # Diff options
    diff_mode: DiffMode = DiffMode.UNIFIED
    context_lines: int = 3
    
    # Persona-specific options
    persona: str = "developer"  # developer, reviewer, compliance

@dataclass
class IRSection:
    """A section of the formatted IR"""
    section_id: str
    section_title: str
    section_content: str
    section_type: str  # header, nodes, edges, metadata, policies
    line_count: int = 0
    
    # Annotations
    annotations: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    
    # Cross-references
    references: List[str] = field(default_factory=list)

@dataclass
class FormattedIR:
    """Complete formatted IR representation"""
    ir_id: str
    plan_id: str
    
    # Formatted content
    formatted_content: str
    sections: List[IRSection] = field(default_factory=list)
    
    # Statistics
    total_lines: int = 0
    total_nodes: int = 0
    total_edges: int = 0
    
    # Formatting metadata
    format_options: PrettyPrintOptions = field(default_factory=PrettyPrintOptions)
    generated_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    generation_time_ms: float = 0.0

@dataclass
class IRDiff:
    """Difference between two IRs"""
    diff_id: str
    source_ir_id: str
    target_ir_id: str
    
    # Diff content
    diff_content: str
    diff_format: DiffMode
    
    # Change statistics
    lines_added: int = 0
    lines_removed: int = 0
    lines_modified: int = 0
    
    # Semantic changes
    nodes_added: List[str] = field(default_factory=list)
    nodes_removed: List[str] = field(default_factory=list)
    nodes_modified: List[str] = field(default_factory=list)
    edges_added: List[str] = field(default_factory=list)
    edges_removed: List[str] = field(default_factory=list)
    
    # Policy changes
    policy_changes: List[str] = field(default_factory=list)
    
    # Metadata
    generated_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

# Task 6.2.42: IR Pretty-Printer Service
class IRPrettyPrinterService:
    """Service for pretty-printing IR for human review"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
        # Formatting templates
        self.templates = self._initialize_templates()
        
        # Keyword highlighting patterns
        self.keyword_patterns = self._initialize_keyword_patterns()
        
        # Persona-specific configurations
        self.persona_configs = self._initialize_persona_configs()
        
        # Cache for formatted IRs
        self.formatted_cache: Dict[str, FormattedIR] = {}
        
        # Statistics
        self.printer_stats = {
            'total_formatted': 0,
            'formats_generated': {fmt.value: 0 for fmt in OutputFormat},
            'diffs_generated': 0,
            'cache_hits': 0,
            'cache_misses': 0,
            'persona_usage': defaultdict(int)
        }
    
    def format_ir(self, ir_data: Dict[str, Any], 
                  options: Optional[PrettyPrintOptions] = None) -> FormattedIR:
        """
        Format IR data into human-readable representation
        
        Args:
            ir_data: IR data to format
            options: Formatting options
            
        Returns:
            FormattedIR with formatted content
        """
        start_time = datetime.now(timezone.utc)
        
        # Use default options if not provided
        if options is None:
            options = PrettyPrintOptions()
        
        # Apply persona-specific configuration
        if options.persona in self.persona_configs:
            persona_config = self.persona_configs[options.persona]
            options = self._apply_persona_config(options, persona_config)
        
        ir_id = ir_data.get('ir_id', f"ir_{hash(str(ir_data))}")
        plan_id = ir_data.get('plan_id', 'unknown')
        
        # Check cache
        cache_key = f"{ir_id}_{hash(str(options))}"
        if cache_key in self.formatted_cache:
            self.printer_stats['cache_hits'] += 1
            return self.formatted_cache[cache_key]
        
        self.printer_stats['cache_misses'] += 1
        
        # Create formatted IR instance
        formatted_ir = FormattedIR(
            ir_id=ir_id,
            plan_id=plan_id,
            format_options=options
        )
        
        # Format different sections
        sections = []
        
        # Header section
        header_section = self._format_header_section(ir_data, options)
        sections.append(header_section)
        
        # Nodes section
        if 'nodes' in ir_data or 'plan_graph' in ir_data:
            nodes_section = self._format_nodes_section(ir_data, options)
            sections.append(nodes_section)
            formatted_ir.total_nodes = len(self._extract_nodes(ir_data))
        
        # Edges section
        if 'edges' in ir_data or 'plan_graph' in ir_data:
            edges_section = self._format_edges_section(ir_data, options)
            sections.append(edges_section)
            formatted_ir.total_edges = len(self._extract_edges(ir_data))
        
        # Policies section
        if options.include_policies and self._has_policies(ir_data):
            policies_section = self._format_policies_section(ir_data, options)
            sections.append(policies_section)
        
        # Metadata section
        if options.include_metadata and 'metadata' in ir_data:
            metadata_section = self._format_metadata_section(ir_data, options)
            sections.append(metadata_section)
        
        # Combine sections
        formatted_ir.sections = sections
        formatted_ir.formatted_content = self._combine_sections(sections, options)
        formatted_ir.total_lines = len(formatted_ir.formatted_content.split('\n'))
        
        # Record generation time
        end_time = datetime.now(timezone.utc)
        formatted_ir.generation_time_ms = (end_time - start_time).total_seconds() * 1000
        
        # Cache result
        self.formatted_cache[cache_key] = formatted_ir
        
        # Update statistics
        self.printer_stats['total_formatted'] += 1
        self.printer_stats['formats_generated'][options.output_format.value] += 1
        self.printer_stats['persona_usage'][options.persona] += 1
        
        self.logger.info(f"✅ Formatted IR: {ir_id} -> {formatted_ir.total_lines} lines ({options.output_format.value})")
        
        return formatted_ir
    
    def generate_ir_diff(self, source_ir: Dict[str, Any], target_ir: Dict[str, Any],
                        options: Optional[PrettyPrintOptions] = None) -> IRDiff:
        """
        Generate diff between two IRs
        
        Args:
            source_ir: Source IR data
            target_ir: Target IR data
            options: Formatting options
            
        Returns:
            IRDiff with difference information
        """
        if options is None:
            options = PrettyPrintOptions()
        
        # Format both IRs
        source_formatted = self.format_ir(source_ir, options)
        target_formatted = self.format_ir(target_ir, options)
        
        diff_id = f"diff_{source_formatted.ir_id}_{target_formatted.ir_id}"
        
        # Create diff instance
        ir_diff = IRDiff(
            diff_id=diff_id,
            source_ir_id=source_formatted.ir_id,
            target_ir_id=target_formatted.ir_id,
            diff_format=options.diff_mode
        )
        
        # Generate text diff
        source_lines = source_formatted.formatted_content.split('\n')
        target_lines = target_formatted.formatted_content.split('\n')
        
        if options.diff_mode == DiffMode.UNIFIED:
            diff_lines = list(difflib.unified_diff(
                source_lines, target_lines,
                fromfile=f"IR {source_formatted.ir_id}",
                tofile=f"IR {target_formatted.ir_id}",
                n=options.context_lines
            ))
        elif options.diff_mode == DiffMode.CONTEXT:
            diff_lines = list(difflib.context_diff(
                source_lines, target_lines,
                fromfile=f"IR {source_formatted.ir_id}",
                tofile=f"IR {target_formatted.ir_id}",
                n=options.context_lines
            ))
        else:
            # Default to unified
            diff_lines = list(difflib.unified_diff(
                source_lines, target_lines,
                n=options.context_lines
            ))
        
        ir_diff.diff_content = '\n'.join(diff_lines)
        
        # Analyze changes
        self._analyze_diff_changes(ir_diff, source_ir, target_ir)
        
        # Update statistics
        self.printer_stats['diffs_generated'] += 1
        
        self.logger.info(f"✅ Generated IR diff: {diff_id} -> +{ir_diff.lines_added}/-{ir_diff.lines_removed}")
        
        return ir_diff
    
    def format_for_persona(self, ir_data: Dict[str, Any], persona: str) -> FormattedIR:
        """
        Format IR for a specific persona
        
        Args:
            ir_data: IR data to format
            persona: Target persona (developer, reviewer, compliance)
            
        Returns:
            FormattedIR tailored for the persona
        """
        options = PrettyPrintOptions(persona=persona)
        return self.format_ir(ir_data, options)
    
    def export_formatted_ir(self, formatted_ir: FormattedIR, 
                           output_format: OutputFormat) -> str:
        """
        Export formatted IR to different output formats
        
        Args:
            formatted_ir: Formatted IR to export
            output_format: Target output format
            
        Returns:
            Exported content as string
        """
        if output_format == OutputFormat.TEXT:
            return formatted_ir.formatted_content
        
        elif output_format == OutputFormat.MARKDOWN:
            return self._convert_to_markdown(formatted_ir)
        
        elif output_format == OutputFormat.HTML:
            return self._convert_to_html(formatted_ir)
        
        elif output_format == OutputFormat.JSON:
            return json.dumps(asdict(formatted_ir), indent=2, default=str)
        
        else:
            return formatted_ir.formatted_content
    
    def get_formatting_statistics(self) -> Dict[str, Any]:
        """Get formatting service statistics"""
        return {
            **self.printer_stats,
            'cached_formats': len(self.formatted_cache),
            'supported_formats': len(OutputFormat),
            'persona_configs': len(self.persona_configs)
        }
    
    def _format_header_section(self, ir_data: Dict[str, Any], 
                              options: PrettyPrintOptions) -> IRSection:
        """Format the header section"""
        lines = []
        
        # Title
        plan_id = ir_data.get('plan_id', 'Unknown Plan')
        ir_version = ir_data.get('version', '1.0')
        
        if options.output_format == OutputFormat.MARKDOWN:
            lines.append(f"# IR: {plan_id}")
            lines.append(f"**Version:** {ir_version}")
        else:
            lines.append("=" * 60)
            lines.append(f"IR: {plan_id}")
            lines.append(f"Version: {ir_version}")
            lines.append("=" * 60)
        
        # Provenance information
        if options.include_provenance and 'provenance' in ir_data:
            provenance = ir_data['provenance']
            lines.append("")
            if options.output_format == OutputFormat.MARKDOWN:
                lines.append("## Provenance")
            else:
                lines.append("Provenance:")
            
            for key, value in provenance.items():
                if options.output_format == OutputFormat.MARKDOWN:
                    lines.append(f"- **{key}:** {value}")
                else:
                    lines.append(f"  {key}: {value}")
        
        # Summary statistics
        nodes = self._extract_nodes(ir_data)
        edges = self._extract_edges(ir_data)
        
        lines.append("")
        if options.output_format == OutputFormat.MARKDOWN:
            lines.append("## Summary")
            lines.append(f"- **Nodes:** {len(nodes)}")
            lines.append(f"- **Edges:** {len(edges)}")
        else:
            lines.append("Summary:")
            lines.append(f"  Nodes: {len(nodes)}")
            lines.append(f"  Edges: {len(edges)}")
        
        return IRSection(
            section_id="header",
            section_title="Header",
            section_content="\n".join(lines),
            section_type="header",
            line_count=len(lines)
        )
    
    def _format_nodes_section(self, ir_data: Dict[str, Any], 
                             options: PrettyPrintOptions) -> IRSection:
        """Format the nodes section"""
        nodes = self._extract_nodes(ir_data)
        lines = []
        
        if options.output_format == OutputFormat.MARKDOWN:
            lines.append("## Nodes")
        else:
            lines.append("")
            lines.append("Nodes:")
            lines.append("-" * 40)
        
        # Filter nodes if specified
        if options.node_types_filter:
            nodes = [n for n in nodes if n.get('type') in options.node_types_filter]
        
        for node in nodes:
            node_lines = self._format_single_node(node, options)
            lines.extend(node_lines)
            lines.append("")
        
        return IRSection(
            section_id="nodes",
            section_title="Nodes",
            section_content="\n".join(lines),
            section_type="nodes",
            line_count=len(lines)
        )
    
    def _format_single_node(self, node: Dict[str, Any], 
                           options: PrettyPrintOptions) -> List[str]:
        """Format a single node"""
        lines = []
        
        node_id = node.get('id', 'unknown')
        node_type = node.get('type', 'unknown')
        
        if options.output_format == OutputFormat.MARKDOWN:
            lines.append(f"### {node_id}")
            lines.append(f"**Type:** {node_type}")
        else:
            lines.append(f"Node: {node_id}")
            lines.append(f"  Type: {node_type}")
        
        # Node description
        if 'description' in node:
            if options.output_format == OutputFormat.MARKDOWN:
                lines.append(f"**Description:** {node['description']}")
            else:
                lines.append(f"  Description: {node['description']}")
        
        # Inputs
        if 'inputs' in node and node['inputs']:
            if options.output_format == OutputFormat.MARKDOWN:
                lines.append("**Inputs:**")
                for inp in node['inputs']:
                    lines.append(f"- {inp.get('name', 'unnamed')}: {inp.get('type', 'unknown')}")
            else:
                lines.append("  Inputs:")
                for inp in node['inputs']:
                    lines.append(f"    - {inp.get('name', 'unnamed')}: {inp.get('type', 'unknown')}")
        
        # Outputs
        if 'outputs' in node and node['outputs']:
            if options.output_format == OutputFormat.MARKDOWN:
                lines.append("**Outputs:**")
                for out in node['outputs']:
                    lines.append(f"- {out.get('name', 'unnamed')}: {out.get('type', 'unknown')}")
            else:
                lines.append("  Outputs:")
                for out in node['outputs']:
                    lines.append(f"    - {out.get('name', 'unnamed')}: {out.get('type', 'unknown')}")
        
        # Policies (if enabled and present)
        if options.include_policies and 'policies' in node:
            policies = node['policies']
            if policies:
                if options.output_format == OutputFormat.MARKDOWN:
                    lines.append("**Policies:**")
                else:
                    lines.append("  Policies:")
                
                for policy_key, policy_value in policies.items():
                    if options.output_format == OutputFormat.MARKDOWN:
                        lines.append(f"- {policy_key}: {policy_value}")
                    else:
                        lines.append(f"    {policy_key}: {policy_value}")
        
        # Trust budget (if present)
        if 'trust_budget' in node:
            trust_budget = node['trust_budget']
            if options.output_format == OutputFormat.MARKDOWN:
                lines.append(f"**Trust Budget:** {trust_budget}")
            else:
                lines.append(f"  Trust Budget: {trust_budget}")
        
        # Fallbacks (if present)
        if 'fallbacks' in node and node['fallbacks']:
            if options.output_format == OutputFormat.MARKDOWN:
                lines.append("**Fallbacks:**")
                for fallback in node['fallbacks']:
                    lines.append(f"- {fallback.get('condition', 'default')} → {fallback.get('target', 'unknown')}")
            else:
                lines.append("  Fallbacks:")
                for fallback in node['fallbacks']:
                    lines.append(f"    {fallback.get('condition', 'default')} → {fallback.get('target', 'unknown')}")
        
        # Comments (if enabled and present)
        if options.include_comments and 'comments' in node:
            comments = node['comments']
            if comments:
                if options.output_format == OutputFormat.MARKDOWN:
                    lines.append("**Comments:**")
                    for comment in comments:
                        lines.append(f"> {comment}")
                else:
                    lines.append("  Comments:")
                    for comment in comments:
                        lines.append(f"    # {comment}")
        
        return lines
    
    def _format_edges_section(self, ir_data: Dict[str, Any], 
                             options: PrettyPrintOptions) -> IRSection:
        """Format the edges section"""
        edges = self._extract_edges(ir_data)
        lines = []
        
        if options.output_format == OutputFormat.MARKDOWN:
            lines.append("## Edges")
            lines.append("| Source | Target | Type | Condition |")
            lines.append("|--------|--------|------|-----------|")
        else:
            lines.append("")
            lines.append("Edges:")
            lines.append("-" * 40)
        
        for edge in edges:
            source = edge.get('source', edge.get('from', 'unknown'))
            target = edge.get('target', edge.get('to', 'unknown'))
            edge_type = edge.get('type', 'data')
            condition = edge.get('condition', '')
            
            if options.output_format == OutputFormat.MARKDOWN:
                lines.append(f"| {source} | {target} | {edge_type} | {condition} |")
            else:
                lines.append(f"  {source} → {target}")
                if edge_type != 'data':
                    lines.append(f"    Type: {edge_type}")
                if condition:
                    lines.append(f"    Condition: {condition}")
        
        return IRSection(
            section_id="edges",
            section_title="Edges",
            section_content="\n".join(lines),
            section_type="edges",
            line_count=len(lines)
        )
    
    def _format_policies_section(self, ir_data: Dict[str, Any], 
                               options: PrettyPrintOptions) -> IRSection:
        """Format the policies section"""
        lines = []
        
        if options.output_format == OutputFormat.MARKDOWN:
            lines.append("## Policies")
        else:
            lines.append("")
            lines.append("Policies:")
            lines.append("-" * 40)
        
        # Extract policies from various sources
        policies = {}
        
        # Global policies
        if 'policies' in ir_data:
            policies.update(ir_data['policies'])
        
        # Node-level policies
        nodes = self._extract_nodes(ir_data)
        for node in nodes:
            if 'policies' in node:
                node_id = node.get('id', 'unknown')
                policies[f"node_{node_id}"] = node['policies']
        
        for policy_key, policy_value in policies.items():
            if options.output_format == OutputFormat.MARKDOWN:
                lines.append(f"### {policy_key}")
                if isinstance(policy_value, dict):
                    for k, v in policy_value.items():
                        lines.append(f"- **{k}:** {v}")
                else:
                    lines.append(f"Value: {policy_value}")
            else:
                lines.append(f"  {policy_key}:")
                if isinstance(policy_value, dict):
                    for k, v in policy_value.items():
                        lines.append(f"    {k}: {v}")
                else:
                    lines.append(f"    Value: {policy_value}")
        
        return IRSection(
            section_id="policies",
            section_title="Policies",
            section_content="\n".join(lines),
            section_type="policies",
            line_count=len(lines)
        )
    
    def _format_metadata_section(self, ir_data: Dict[str, Any], 
                               options: PrettyPrintOptions) -> IRSection:
        """Format the metadata section"""
        metadata = ir_data.get('metadata', {})
        lines = []
        
        if options.output_format == OutputFormat.MARKDOWN:
            lines.append("## Metadata")
        else:
            lines.append("")
            lines.append("Metadata:")
            lines.append("-" * 40)
        
        for key, value in metadata.items():
            if options.output_format == OutputFormat.MARKDOWN:
                lines.append(f"- **{key}:** {value}")
            else:
                lines.append(f"  {key}: {value}")
        
        return IRSection(
            section_id="metadata",
            section_title="Metadata",
            section_content="\n".join(lines),
            section_type="metadata",
            line_count=len(lines)
        )
    
    def _combine_sections(self, sections: List[IRSection], 
                         options: PrettyPrintOptions) -> str:
        """Combine sections into final formatted content"""
        content_parts = []
        
        for section in sections:
            content_parts.append(section.section_content)
        
        combined = "\n\n".join(content_parts)
        
        # Apply syntax highlighting if specified
        if options.highlight_keywords:
            combined = self._apply_keyword_highlighting(combined, options.highlight_keywords)
        
        return combined
    
    def _apply_keyword_highlighting(self, content: str, keywords: List[str]) -> str:
        """Apply keyword highlighting to content"""
        # For text output, use simple markers
        # In a real implementation, this might use ANSI colors or markup
        for keyword in keywords:
            pattern = re.compile(r'\b' + re.escape(keyword) + r'\b', re.IGNORECASE)
            content = pattern.sub(f"**{keyword}**", content)
        
        return content
    
    def _convert_to_markdown(self, formatted_ir: FormattedIR) -> str:
        """Convert formatted IR to Markdown"""
        # If already in markdown format, return as-is
        if formatted_ir.format_options.output_format == OutputFormat.MARKDOWN:
            return formatted_ir.formatted_content
        
        # Convert text to markdown
        lines = formatted_ir.formatted_content.split('\n')
        markdown_lines = []
        
        for line in lines:
            # Convert headers
            if line.startswith('='):
                continue  # Skip separator lines
            elif line and not line.startswith(' '):
                # Treat as header
                markdown_lines.append(f"## {line}")
            else:
                markdown_lines.append(line)
        
        return '\n'.join(markdown_lines)
    
    def _convert_to_html(self, formatted_ir: FormattedIR) -> str:
        """Convert formatted IR to HTML"""
        html_parts = [
            "<!DOCTYPE html>",
            "<html>",
            "<head>",
            f"<title>IR: {formatted_ir.plan_id}</title>",
            "<style>",
            "body { font-family: monospace; margin: 20px; }",
            ".section { margin-bottom: 20px; }",
            ".node { border: 1px solid #ccc; padding: 10px; margin: 10px 0; }",
            ".policy { background-color: #f0f0f0; padding: 5px; }",
            "</style>",
            "</head>",
            "<body>"
        ]
        
        for section in formatted_ir.sections:
            html_parts.append(f'<div class="section">')
            html_parts.append(f'<h2>{section.section_title}</h2>')
            
            # Convert content to HTML
            content_html = section.section_content.replace('\n', '<br>\n')
            content_html = content_html.replace('  ', '&nbsp;&nbsp;')
            
            html_parts.append(f'<pre>{content_html}</pre>')
            html_parts.append('</div>')
        
        html_parts.extend(["</body>", "</html>"])
        
        return '\n'.join(html_parts)
    
    def _analyze_diff_changes(self, ir_diff: IRDiff, source_ir: Dict[str, Any], target_ir: Dict[str, Any]):
        """Analyze semantic changes in the diff"""
        # Count line changes
        diff_lines = ir_diff.diff_content.split('\n')
        for line in diff_lines:
            if line.startswith('+') and not line.startswith('+++'):
                ir_diff.lines_added += 1
            elif line.startswith('-') and not line.startswith('---'):
                ir_diff.lines_removed += 1
        
        # Analyze node changes
        source_nodes = {n.get('id'): n for n in self._extract_nodes(source_ir)}
        target_nodes = {n.get('id'): n for n in self._extract_nodes(target_ir)}
        
        # Find added/removed/modified nodes
        source_node_ids = set(source_nodes.keys())
        target_node_ids = set(target_nodes.keys())
        
        ir_diff.nodes_added = list(target_node_ids - source_node_ids)
        ir_diff.nodes_removed = list(source_node_ids - target_node_ids)
        
        # Check for modified nodes
        common_nodes = source_node_ids & target_node_ids
        for node_id in common_nodes:
            if source_nodes[node_id] != target_nodes[node_id]:
                ir_diff.nodes_modified.append(node_id)
        
        # Analyze edge changes
        source_edges = self._extract_edges(source_ir)
        target_edges = self._extract_edges(target_ir)
        
        source_edge_sigs = [f"{e.get('source', e.get('from'))}→{e.get('target', e.get('to'))}" for e in source_edges]
        target_edge_sigs = [f"{e.get('source', e.get('from'))}→{e.get('target', e.get('to'))}" for e in target_edges]
        
        ir_diff.edges_added = list(set(target_edge_sigs) - set(source_edge_sigs))
        ir_diff.edges_removed = list(set(source_edge_sigs) - set(target_edge_sigs))
    
    def _extract_nodes(self, ir_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Extract nodes from IR data"""
        if 'nodes' in ir_data:
            return ir_data['nodes']
        elif 'plan_graph' in ir_data and 'nodes' in ir_data['plan_graph']:
            return ir_data['plan_graph']['nodes']
        else:
            return []
    
    def _extract_edges(self, ir_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Extract edges from IR data"""
        if 'edges' in ir_data:
            return ir_data['edges']
        elif 'plan_graph' in ir_data and 'edges' in ir_data['plan_graph']:
            return ir_data['plan_graph']['edges']
        else:
            return []
    
    def _has_policies(self, ir_data: Dict[str, Any]) -> bool:
        """Check if IR data has policy information"""
        if 'policies' in ir_data:
            return True
        
        # Check nodes for policies
        nodes = self._extract_nodes(ir_data)
        return any('policies' in node for node in nodes)
    
    def _apply_persona_config(self, options: PrettyPrintOptions, 
                            persona_config: Dict[str, Any]) -> PrettyPrintOptions:
        """Apply persona-specific configuration to options"""
        for key, value in persona_config.items():
            if hasattr(options, key):
                setattr(options, key, value)
        
        return options
    
    def _initialize_templates(self) -> Dict[str, str]:
        """Initialize formatting templates"""
        return {
            'node_header': "Node: {id} ({type})",
            'edge_format': "{source} → {target}",
            'policy_format': "{key}: {value}",
            'section_separator': "-" * 40
        }
    
    def _initialize_keyword_patterns(self) -> Dict[str, str]:
        """Initialize keyword highlighting patterns"""
        return {
            'ml_keywords': r'\b(ml_node|predict|classify|score|confidence)\b',
            'policy_keywords': r'\b(policy|trust|fallback|residency)\b',
            'control_keywords': r'\b(if|then|else|while|for|condition)\b'
        }
    
    def _initialize_persona_configs(self) -> Dict[str, Dict[str, Any]]:
        """Initialize persona-specific configurations"""
        return {
            'developer': {
                'annotation_level': AnnotationLevel.VERBOSE,
                'include_comments': True,
                'include_metadata': True,
                'output_format': OutputFormat.TEXT
            },
            'reviewer': {
                'annotation_level': AnnotationLevel.STANDARD,
                'include_policies': True,
                'include_provenance': True,
                'output_format': OutputFormat.MARKDOWN
            },
            'compliance': {
                'annotation_level': AnnotationLevel.VERBOSE,
                'include_policies': True,
                'include_provenance': True,
                'include_metadata': True,
                'output_format': OutputFormat.HTML
            }
        }

# API Interface
class IRPrettyPrinterAPI:
    """API interface for IR pretty-printing operations"""
    
    def __init__(self, printer_service: Optional[IRPrettyPrinterService] = None):
        self.printer_service = printer_service or IRPrettyPrinterService()
    
    def format_ir(self, ir_data: Dict[str, Any], 
                  format_options: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """API endpoint to format IR"""
        try:
            options = None
            if format_options:
                options = PrettyPrintOptions(**format_options)
            
            formatted_ir = self.printer_service.format_ir(ir_data, options)
            
            return {
                'success': True,
                'formatted_ir': asdict(formatted_ir)
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': str(e),
                'error_type': type(e).__name__
            }
    
    def generate_diff(self, source_ir: Dict[str, Any], target_ir: Dict[str, Any],
                     diff_options: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """API endpoint to generate IR diff"""
        try:
            options = None
            if diff_options:
                options = PrettyPrintOptions(**diff_options)
            
            ir_diff = self.printer_service.generate_ir_diff(source_ir, target_ir, options)
            
            return {
                'success': True,
                'diff': asdict(ir_diff)
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': str(e)
            }
    
    def format_for_persona(self, ir_data: Dict[str, Any], persona: str) -> Dict[str, Any]:
        """API endpoint to format IR for specific persona"""
        try:
            formatted_ir = self.printer_service.format_for_persona(ir_data, persona)
            
            return {
                'success': True,
                'formatted_ir': asdict(formatted_ir)
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': str(e)
            }

# Test Functions
def create_test_ir_data() -> Dict[str, Any]:
    """Create test IR data for formatting"""
    return {
        'ir_id': 'test_ir_001',
        'plan_id': 'test_plan_formatting',
        'version': '1.0.0',
        'provenance': {
            'author': 'test_user',
            'created_at': '2024-01-01T00:00:00Z',
            'source_file': 'test_plan.rbia',
            'commit_sha': 'abc123def456'
        },
        'plan_graph': {
            'nodes': [
                {
                    'id': 'start_node',
                    'type': 'start',
                    'description': 'Starting point of the workflow',
                    'outputs': [
                        {'name': 'trigger', 'type': 'event'}
                    ],
                    'comments': ['This is the entry point'],
                    'policies': {
                        'residency': 'us-east-1',
                        'trust_level': 'high'
                    }
                },
                {
                    'id': 'ml_classifier',
                    'type': 'ml_node',
                    'description': 'ML classification node',
                    'inputs': [
                        {'name': 'features', 'type': 'vector'},
                        {'name': 'model_id', 'type': 'string'}
                    ],
                    'outputs': [
                        {'name': 'prediction', 'type': 'string'},
                        {'name': 'confidence', 'type': 'float'}
                    ],
                    'trust_budget': {
                        'min_confidence': 0.8,
                        'auto_execute_threshold': 0.9
                    },
                    'fallbacks': [
                        {
                            'condition': 'confidence < 0.8',
                            'target': 'manual_review',
                            'reason': 'Low confidence prediction'
                        }
                    ],
                    'policies': {
                        'model_governance': 'strict',
                        'explainability': 'required'
                    }
                },
                {
                    'id': 'decision_node',
                    'type': 'decision',
                    'description': 'Decision based on ML output',
                    'inputs': [
                        {'name': 'prediction', 'type': 'string'},
                        {'name': 'confidence', 'type': 'float'}
                    ],
                    'outputs': [
                        {'name': 'action', 'type': 'string'}
                    ]
                }
            ],
            'edges': [
                {
                    'source': 'start_node',
                    'target': 'ml_classifier',
                    'type': 'data'
                },
                {
                    'source': 'ml_classifier',
                    'target': 'decision_node',
                    'type': 'data',
                    'condition': 'confidence >= 0.8'
                }
            ]
        },
        'policies': {
            'global_timeout': '300s',
            'error_handling': 'fail_fast',
            'audit_level': 'full'
        },
        'metadata': {
            'environment': 'production',
            'tenant_id': 'test_tenant',
            'cost_estimate': '$0.50',
            'sla_target': '99.9%'
        }
    }

def run_ir_pretty_printer_tests():
    """Run comprehensive IR pretty-printer tests"""
    print("=== IR Pretty-Printer Service Tests ===")
    
    # Initialize service
    printer_service = IRPrettyPrinterService()
    printer_api = IRPrettyPrinterAPI(printer_service)
    
    # Create test IR data
    test_ir = create_test_ir_data()
    
    # Test 1: Basic formatting
    print("\n1. Testing basic IR formatting...")
    
    # Test different output formats
    formats = [OutputFormat.TEXT, OutputFormat.MARKDOWN, OutputFormat.HTML]
    
    for output_format in formats:
        options = PrettyPrintOptions(output_format=output_format)
        formatted_ir = printer_service.format_ir(test_ir, options)
        
        print(f"   {output_format.value} format:")
        print(f"     Total lines: {formatted_ir.total_lines}")
        print(f"     Sections: {len(formatted_ir.sections)}")
        print(f"     Generation time: {formatted_ir.generation_time_ms:.1f}ms")
        
        # Show a snippet of the formatted content
        lines = formatted_ir.formatted_content.split('\n')
        print(f"     Preview: {lines[0][:50]}...")
    
    # Test 2: Persona-specific formatting
    print("\n2. Testing persona-specific formatting...")
    
    personas = ['developer', 'reviewer', 'compliance']
    
    for persona in personas:
        formatted_ir = printer_service.format_for_persona(test_ir, persona)
        
        print(f"   {persona} format:")
        print(f"     Output format: {formatted_ir.format_options.output_format.value}")
        print(f"     Annotation level: {formatted_ir.format_options.annotation_level.value}")
        print(f"     Include policies: {formatted_ir.format_options.include_policies}")
        print(f"     Total lines: {formatted_ir.total_lines}")
    
    # Test 3: Different annotation levels
    print("\n3. Testing annotation levels...")
    
    annotation_levels = [AnnotationLevel.MINIMAL, AnnotationLevel.STANDARD, AnnotationLevel.VERBOSE]
    
    for level in annotation_levels:
        options = PrettyPrintOptions(annotation_level=level)
        formatted_ir = printer_service.format_ir(test_ir, options)
        
        print(f"   {level.value} annotation:")
        print(f"     Total lines: {formatted_ir.total_lines}")
        print(f"     Sections: {len(formatted_ir.sections)}")
    
    # Test 4: IR diff generation
    print("\n4. Testing IR diff generation...")
    
    # Create a modified version of the test IR
    modified_ir = json.loads(json.dumps(test_ir))
    modified_ir['plan_graph']['nodes'].append({
        'id': 'new_node',
        'type': 'action',
        'description': 'Newly added node'
    })
    modified_ir['plan_graph']['nodes'][1]['description'] = 'Modified ML classification node'
    
    # Generate diff
    ir_diff = printer_service.generate_ir_diff(test_ir, modified_ir)
    
    print(f"   IR diff generated: {ir_diff.diff_id}")
    print(f"     Lines added: {ir_diff.lines_added}")
    print(f"     Lines removed: {ir_diff.lines_removed}")
    print(f"     Nodes added: {len(ir_diff.nodes_added)}")
    print(f"     Nodes modified: {len(ir_diff.nodes_modified)}")
    
    # Show diff snippet
    diff_lines = ir_diff.diff_content.split('\n')
    print(f"     Diff preview: {len(diff_lines)} lines")
    
    # Test 5: Export to different formats
    print("\n5. Testing export to different formats...")
    
    base_formatted = printer_service.format_ir(test_ir)
    
    for export_format in [OutputFormat.MARKDOWN, OutputFormat.HTML, OutputFormat.JSON]:
        exported_content = printer_service.export_formatted_ir(base_formatted, export_format)
        
        print(f"   {export_format.value} export:")
        print(f"     Content length: {len(exported_content)} characters")
        print(f"     Preview: {exported_content[:100].replace(chr(10), ' ')}...")
    
    # Test 6: Filtering and options
    print("\n6. Testing filtering and options...")
    
    # Test node type filtering
    filter_options = PrettyPrintOptions(
        node_types_filter=['ml_node', 'decision'],
        include_comments=True,
        highlight_keywords=['ml_node', 'confidence']
    )
    
    filtered_ir = printer_service.format_ir(test_ir, filter_options)
    
    print(f"   Filtered formatting:")
    print(f"     Total lines: {filtered_ir.total_lines}")
    print(f"     Filtered for ML and decision nodes")
    print(f"     Keywords highlighted: {filter_options.highlight_keywords}")
    
    # Test 7: Section analysis
    print("\n7. Testing section analysis...")
    
    verbose_options = PrettyPrintOptions(annotation_level=AnnotationLevel.VERBOSE)
    verbose_ir = printer_service.format_ir(test_ir, verbose_options)
    
    print(f"   Section breakdown:")
    for section in verbose_ir.sections:
        print(f"     {section.section_title}: {section.line_count} lines")
        if section.annotations:
            print(f"       Annotations: {len(section.annotations)}")
    
    # Test 8: API interface
    print("\n8. Testing API interface...")
    
    # Test API formatting
    api_format_result = printer_api.format_ir(test_ir, {
        'output_format': 'markdown',
        'annotation_level': 'standard'
    })
    print(f"   API formatting: {'✅ PASS' if api_format_result['success'] else '❌ FAIL'}")
    
    # Test API diff
    api_diff_result = printer_api.generate_diff(test_ir, modified_ir, {
        'diff_mode': 'unified',
        'context_lines': 3
    })
    print(f"   API diff: {'✅ PASS' if api_diff_result['success'] else '❌ FAIL'}")
    
    # Test API persona formatting
    api_persona_result = printer_api.format_for_persona(test_ir, 'compliance')
    print(f"   API persona formatting: {'✅ PASS' if api_persona_result['success'] else '❌ FAIL'}")
    
    # Test 9: Edge cases
    print("\n9. Testing edge cases...")
    
    # Empty IR
    empty_ir = {'ir_id': 'empty', 'plan_id': 'empty_plan', 'plan_graph': {'nodes': [], 'edges': []}}
    empty_formatted = printer_service.format_ir(empty_ir)
    print(f"   Empty IR: {empty_formatted.total_lines} lines, {len(empty_formatted.sections)} sections")
    
    # IR with no metadata
    minimal_ir = {'ir_id': 'minimal', 'plan_id': 'minimal_plan'}
    minimal_formatted = printer_service.format_ir(minimal_ir)
    print(f"   Minimal IR: {minimal_formatted.total_lines} lines")
    
    # Test 10: Performance and caching
    print("\n10. Testing performance and caching...")
    
    # Format the same IR multiple times to test caching
    start_time = datetime.now(timezone.utc)
    for i in range(5):
        cached_formatted = printer_service.format_ir(test_ir)
    end_time = datetime.now(timezone.utc)
    
    total_time = (end_time - start_time).total_seconds() * 1000
    print(f"   5 format operations: {total_time:.1f}ms total")
    
    # Test statistics
    stats = printer_service.get_formatting_statistics()
    print(f"   Cache hits: {stats['cache_hits']}")
    print(f"   Cache misses: {stats['cache_misses']}")
    print(f"   Total formatted: {stats['total_formatted']}")
    print(f"   Diffs generated: {stats['diffs_generated']}")
    
    print(f"\n=== Test Summary ===")
    final_stats = printer_service.get_formatting_statistics()
    print(f"IR pretty-printer service tested successfully")
    print(f"Total formatted: {final_stats['total_formatted']}")
    print(f"Formats generated: {final_stats['formats_generated']}")
    print(f"Diffs generated: {final_stats['diffs_generated']}")
    print(f"Cache hit rate: {final_stats['cache_hits']}/{final_stats['cache_hits'] + final_stats['cache_misses']}")
    print(f"Persona usage: {dict(final_stats['persona_usage'])}")
    
    return printer_service, printer_api

if __name__ == "__main__":
    run_ir_pretty_printer_tests()
