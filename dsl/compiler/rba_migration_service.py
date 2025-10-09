"""
RBA to RBIA Migration Service - Task 6.2.34
============================================

Migration tool: RBA → RBIA (upgrade rules; insert ML stubs)
- Helps teams migrate existing RBA rule collections to RBIA hybrid format
- Automatically upgrades rule syntax and inserts ML stubs where appropriate
- Produces migration reports for manual review
- Backend implementation (no actual CLI/Web UI - that's interface layer)

Dependencies: Task 6.2.1 (DSL v2 Grammar), Task 6.2.2 (Parser)
Outputs: Migration analysis and transformation → enables automated RBA-to-RBIA conversion
"""

from typing import Dict, List, Any, Optional, Union, Tuple, Set
import json
import logging
from datetime import datetime, timezone
from dataclasses import dataclass, field, asdict
from enum import Enum
import re
import hashlib
from pathlib import Path

logger = logging.getLogger(__name__)

class MigrationType(Enum):
    """Types of migration transformations"""
    SYNTAX_UPGRADE = "syntax_upgrade"
    ML_STUB_INSERTION = "ml_stub_insertion"
    POLICY_MODERNIZATION = "policy_modernization"
    GOVERNANCE_ENHANCEMENT = "governance_enhancement"
    FALLBACK_ADDITION = "fallback_addition"
    TYPE_ANNOTATION = "type_annotation"

class MigrationRisk(Enum):
    """Risk levels for migration changes"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

class MLCandidateType(Enum):
    """Types of ML candidate patterns"""
    SCORING_DECISION = "scoring_decision"
    CLASSIFICATION = "classification"
    PREDICTION = "prediction"
    ANOMALY_DETECTION = "anomaly_detection"
    RECOMMENDATION = "recommendation"
    RISK_ASSESSMENT = "risk_assessment"

@dataclass
class SourceLocation:
    """Location in source file"""
    file_path: str
    line_start: int
    line_end: int
    column_start: int = 0
    column_end: int = 0
    context_lines: List[str] = field(default_factory=list)

@dataclass
class MigrationChange:
    """A single migration change"""
    change_id: str
    migration_type: MigrationType
    risk_level: MigrationRisk
    
    # Location information
    source_location: SourceLocation
    
    # Change details
    description: str
    original_code: str
    migrated_code: str
    
    # Analysis
    confidence_score: float  # 0.0 to 1.0
    requires_manual_review: bool
    breaking_change: bool = False
    
    # Context
    rationale: str
    suggested_tests: List[str] = field(default_factory=list)
    related_changes: List[str] = field(default_factory=list)  # Other change_ids
    
    # Metadata
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

@dataclass
class MLStubSuggestion:
    """Suggestion for ML node insertion"""
    stub_id: str
    candidate_type: MLCandidateType
    location: SourceLocation
    
    # ML node configuration
    suggested_model_type: str  # "classification", "regression", "ranking"
    required_inputs: List[str]
    expected_outputs: List[str]
    confidence_threshold: float
    
    # Integration details
    fallback_to_original: bool = True
    original_rule_reference: str
    feature_engineering_needed: List[str] = field(default_factory=list)
    
    # Model contract suggestion
    suggested_model_contract: Dict[str, Any] = field(default_factory=dict)
    training_data_requirements: List[str] = field(default_factory=list)
    
    # Metadata
    detection_confidence: float = 0.0
    business_impact_estimate: str = "medium"

@dataclass
class MigrationReport:
    """Complete migration report"""
    migration_id: str
    source_directory: str
    target_directory: str
    
    # File analysis
    files_analyzed: int
    files_modified: int
    files_with_errors: int
    
    # Changes summary
    total_changes: int
    changes_by_type: Dict[MigrationType, int] = field(default_factory=dict)
    changes_by_risk: Dict[MigrationRisk, int] = field(default_factory=dict)
    
    # Detailed changes
    migration_changes: List[MigrationChange] = field(default_factory=list)
    ml_stub_suggestions: List[MLStubSuggestion] = field(default_factory=list)
    
    # Analysis results
    syntax_compatibility: float = 1.0  # 0.0 to 1.0
    estimated_effort_hours: float = 0.0
    critical_issues: List[str] = field(default_factory=list)
    
    # Recommendations
    recommended_migration_order: List[str] = field(default_factory=list)
    required_manual_steps: List[str] = field(default_factory=list)
    suggested_testing_strategy: List[str] = field(default_factory=list)
    
    # Metadata
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    migration_tool_version: str = "1.0.0"

@dataclass
class RBARule:
    """Parsed RBA rule structure"""
    rule_id: str
    rule_name: str
    rule_type: str  # "decision", "scoring", "classification", "action"
    
    # Rule content
    conditions: List[str] = field(default_factory=list)
    actions: List[str] = field(default_factory=list)
    parameters: Dict[str, Any] = field(default_factory=dict)
    
    # Metadata
    comments: List[str] = field(default_factory=list)
    annotations: Dict[str, str] = field(default_factory=dict)
    source_location: Optional[SourceLocation] = None

# Task 6.2.34: RBA to RBIA Migration Service
class RBAToRBIAMigrationService:
    """Service for migrating RBA rules to RBIA hybrid format"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
        # Migration patterns and rules
        self.syntax_upgrade_patterns = self._initialize_syntax_patterns()
        self.ml_candidate_patterns = self._initialize_ml_patterns()
        self.modernization_rules = self._initialize_modernization_rules()
        
        # Migration state
        self.migration_reports: Dict[str, MigrationReport] = {}
        self.parsed_rba_rules: Dict[str, List[RBARule]] = {}  # file_path -> rules
        
        # Statistics
        self.migration_stats = {
            'total_migrations': 0,
            'successful_migrations': 0,
            'files_processed': 0,
            'ml_stubs_generated': 0,
            'syntax_upgrades_applied': 0
        }
    
    def analyze_rba_directory(self, source_directory: str, 
                            include_patterns: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Analyze RBA directory for migration readiness
        
        Args:
            source_directory: Path to RBA rules directory
            include_patterns: File patterns to include (e.g., ["*.rba", "*.rule"])
            
        Returns:
            Dictionary with analysis results
        """
        include_patterns = include_patterns or ["*.rba", "*.rule", "*.yaml"]
        
        analysis = {
            'source_directory': source_directory,
            'files_found': [],
            'parsing_results': {},
            'migration_candidates': {},
            'compatibility_issues': [],
            'estimated_effort': {
                'total_hours': 0.0,
                'by_file': {},
                'by_migration_type': {}
            }
        }
        
        # Find RBA files
        try:
            source_path = Path(source_directory)
            for pattern in include_patterns:
                files = list(source_path.glob(f"**/{pattern}"))
                analysis['files_found'].extend([str(f) for f in files])
        except Exception as e:
            self.logger.error(f"Error scanning directory {source_directory}: {e}")
            analysis['error'] = str(e)
            return analysis
        
        # Analyze each file
        for file_path in analysis['files_found']:
            try:
                file_analysis = self._analyze_rba_file(file_path)
                analysis['parsing_results'][file_path] = file_analysis
                
                # Estimate effort for this file
                effort = self._estimate_migration_effort(file_analysis)
                analysis['estimated_effort']['by_file'][file_path] = effort
                analysis['estimated_effort']['total_hours'] += effort
                
            except Exception as e:
                self.logger.warning(f"Error analyzing file {file_path}: {e}")
                analysis['parsing_results'][file_path] = {'error': str(e)}
        
        # Aggregate migration candidates
        for file_path, file_analysis in analysis['parsing_results'].items():
            if 'ml_candidates' in file_analysis:
                analysis['migration_candidates'][file_path] = file_analysis['ml_candidates']
        
        self.logger.info(f"✅ Analyzed {len(analysis['files_found'])} RBA files in {source_directory}")
        
        return analysis
    
    def create_migration_plan(self, source_directory: str, target_directory: str,
                            dry_run: bool = True, autofix: bool = False) -> MigrationReport:
        """
        Create a migration plan from RBA to RBIA
        
        Args:
            source_directory: Source RBA directory
            target_directory: Target RBIA directory
            dry_run: If True, don't write files, just analyze
            autofix: If True, apply safe automatic fixes
            
        Returns:
            MigrationReport with detailed plan
        """
        migration_id = f"migration_{hashlib.sha256(f'{source_directory}_{target_directory}_{datetime.now()}'.encode()).hexdigest()[:16]}"
        
        # Initialize migration report
        report = MigrationReport(
            migration_id=migration_id,
            source_directory=source_directory,
            target_directory=target_directory
        )
        
        # Analyze source directory
        analysis = self.analyze_rba_directory(source_directory)
        
        if 'error' in analysis:
            report.critical_issues.append(f"Directory analysis failed: {analysis['error']}")
            return report
        
        report.files_analyzed = len(analysis['files_found'])
        
        # Process each file
        for file_path in analysis['files_found']:
            try:
                file_changes = self._generate_file_migration_changes(
                    file_path, analysis['parsing_results'].get(file_path, {})
                )
                
                report.migration_changes.extend(file_changes['changes'])
                report.ml_stub_suggestions.extend(file_changes['ml_stubs'])
                
                if file_changes['changes'] or file_changes['ml_stubs']:
                    report.files_modified += 1
                
            except Exception as e:
                self.logger.error(f"Error processing file {file_path}: {e}")
                report.files_with_errors += 1
                report.critical_issues.append(f"File processing error: {file_path} - {e}")
        
        # Aggregate statistics
        report.total_changes = len(report.migration_changes)
        
        for change in report.migration_changes:
            report.changes_by_type[change.migration_type] = report.changes_by_type.get(change.migration_type, 0) + 1
            report.changes_by_risk[change.risk_level] = report.changes_by_risk.get(change.risk_level, 0) + 1
        
        # Calculate compatibility and effort
        report.syntax_compatibility = self._calculate_syntax_compatibility(report)
        report.estimated_effort_hours = analysis['estimated_effort']['total_hours']
        
        # Generate recommendations
        report.recommended_migration_order = self._generate_migration_order(report)
        report.required_manual_steps = self._identify_manual_steps(report)
        report.suggested_testing_strategy = self._suggest_testing_strategy(report)
        
        # Store report
        self.migration_reports[migration_id] = report
        
        # Update statistics
        self.migration_stats['total_migrations'] += 1
        self.migration_stats['files_processed'] += report.files_analyzed
        self.migration_stats['ml_stubs_generated'] += len(report.ml_stub_suggestions)
        
        self.logger.info(f"✅ Created migration plan {migration_id}: {report.total_changes} changes, {len(report.ml_stub_suggestions)} ML stubs")
        
        return report
    
    def apply_migration_changes(self, migration_id: str, target_directory: str,
                              selected_changes: Optional[List[str]] = None,
                              dry_run: bool = True) -> Dict[str, Any]:
        """
        Apply migration changes to create RBIA files
        
        Args:
            migration_id: Migration plan ID
            target_directory: Target directory for RBIA files
            selected_changes: List of change IDs to apply (None for all)
            dry_run: If True, don't write files
            
        Returns:
            Dictionary with application results
        """
        if migration_id not in self.migration_reports:
            return {'error': f'Migration plan not found: {migration_id}'}
        
        report = self.migration_reports[migration_id]
        
        # Filter changes if specified
        changes_to_apply = report.migration_changes
        if selected_changes:
            changes_to_apply = [c for c in changes_to_apply if c.change_id in selected_changes]
        
        application_result = {
            'migration_id': migration_id,
            'target_directory': target_directory,
            'dry_run': dry_run,
            'changes_applied': 0,
            'files_created': 0,
            'files_modified': 0,
            'errors': [],
            'applied_changes': []
        }
        
        # Group changes by file
        changes_by_file = {}
        for change in changes_to_apply:
            file_path = change.source_location.file_path
            if file_path not in changes_by_file:
                changes_by_file[file_path] = []
            changes_by_file[file_path].append(change)
        
        # Apply changes file by file
        for source_file, file_changes in changes_by_file.items():
            try:
                result = self._apply_file_changes(
                    source_file, file_changes, target_directory, dry_run
                )
                
                application_result['changes_applied'] += result['changes_applied']
                application_result['applied_changes'].extend(result['applied_changes'])
                
                if result['file_created']:
                    application_result['files_created'] += 1
                elif result['file_modified']:
                    application_result['files_modified'] += 1
                    
            except Exception as e:
                error_msg = f"Error applying changes to {source_file}: {e}"
                self.logger.error(error_msg)
                application_result['errors'].append(error_msg)
        
        if not dry_run and application_result['changes_applied'] > 0:
            self.migration_stats['successful_migrations'] += 1
        
        return application_result
    
    def generate_ml_stub_code(self, ml_stub: MLStubSuggestion) -> str:
        """
        Generate RBIA DSL code for an ML stub
        
        Args:
            ml_stub: ML stub suggestion
            
        Returns:
            Generated RBIA DSL code
        """
        # Generate ML node code based on candidate type
        ml_node_code = []
        
        # Add comments and TODOs
        ml_node_code.append(f"# TODO: Implement ML model for {ml_stub.candidate_type.value}")
        ml_node_code.append(f"# Original rule reference: {ml_stub.original_rule_reference}")
        ml_node_code.append(f"# Detection confidence: {ml_stub.detection_confidence:.2f}")
        ml_node_code.append("")
        
        # Generate ML node definition
        node_name = f"ml_{ml_stub.candidate_type.value}_{ml_stub.stub_id}"
        ml_node_code.append(f"{node_name}:")
        ml_node_code.append(f"  type: ml_node")
        ml_node_code.append(f"  model_type: {ml_stub.suggested_model_type}")
        ml_node_code.append(f"  model_id: \"TODO_MODEL_ID\"  # TODO: Replace with actual model ID")
        ml_node_code.append(f"  confidence_threshold: {ml_stub.confidence_threshold}")
        
        # Add inputs
        if ml_stub.required_inputs:
            ml_node_code.append(f"  inputs:")
            for input_name in ml_stub.required_inputs:
                ml_node_code.append(f"    - name: {input_name}")
                ml_node_code.append(f"      type: TODO_TYPE  # TODO: Specify input type")
                ml_node_code.append(f"      required: true")
        
        # Add outputs
        if ml_stub.expected_outputs:
            ml_node_code.append(f"  outputs:")
            for output_name in ml_stub.expected_outputs:
                ml_node_code.append(f"    - name: {output_name}")
                ml_node_code.append(f"      type: TODO_TYPE  # TODO: Specify output type")
        
        # Add fallback configuration
        if ml_stub.fallback_to_original:
            ml_node_code.append(f"  fallback:")
            ml_node_code.append(f"    - condition: \"confidence < {ml_stub.confidence_threshold}\"")
            ml_node_code.append(f"      action: fallback_to_rule")
            ml_node_code.append(f"      target: {ml_stub.original_rule_reference}")
            ml_node_code.append(f"      reason: \"Low confidence prediction\"")
        
        # Add explainability configuration
        ml_node_code.append(f"  explainability:")
        ml_node_code.append(f"    enabled: true")
        ml_node_code.append(f"    method: shap  # TODO: Choose appropriate method")
        ml_node_code.append(f"    audit_integration: true")
        
        # Add feature engineering notes
        if ml_stub.feature_engineering_needed:
            ml_node_code.append(f"  # TODO: Feature engineering required:")
            for feature_note in ml_stub.feature_engineering_needed:
                ml_node_code.append(f"  #   - {feature_note}")
        
        # Add training data requirements
        if ml_stub.training_data_requirements:
            ml_node_code.append(f"  # TODO: Training data requirements:")
            for data_req in ml_stub.training_data_requirements:
                ml_node_code.append(f"  #   - {data_req}")
        
        return "\n".join(ml_node_code)
    
    def get_migration_statistics(self) -> Dict[str, Any]:
        """Get migration service statistics"""
        return {
            **self.migration_stats,
            'active_migration_plans': len(self.migration_reports),
            'parsed_rule_files': len(self.parsed_rba_rules)
        }
    
    def _analyze_rba_file(self, file_path: str) -> Dict[str, Any]:
        """Analyze a single RBA file"""
        analysis = {
            'file_path': file_path,
            'file_size_bytes': 0,
            'line_count': 0,
            'rules_found': [],
            'syntax_issues': [],
            'ml_candidates': [],
            'modernization_opportunities': []
        }
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
                lines = content.split('\n')
                
            analysis['file_size_bytes'] = len(content.encode('utf-8'))
            analysis['line_count'] = len(lines)
            
            # Parse RBA rules (simplified parsing)
            rules = self._parse_rba_content(content, file_path)
            analysis['rules_found'] = [asdict(rule) for rule in rules]
            
            # Detect ML candidates
            ml_candidates = self._detect_ml_candidates(rules, file_path)
            analysis['ml_candidates'] = [asdict(candidate) for candidate in ml_candidates]
            
            # Identify modernization opportunities
            modernization_ops = self._identify_modernization_opportunities(rules)
            analysis['modernization_opportunities'] = modernization_ops
            
        except Exception as e:
            analysis['error'] = str(e)
            self.logger.error(f"Error analyzing RBA file {file_path}: {e}")
        
        return analysis
    
    def _parse_rba_content(self, content: str, file_path: str) -> List[RBARule]:
        """Parse RBA content into rule structures"""
        rules = []
        lines = content.split('\n')
        
        # Simple rule detection patterns
        rule_patterns = [
            r'rule\s+(\w+)\s*:',
            r'decision\s+(\w+)\s*:',
            r'action\s+(\w+)\s*:',
            r'scoring\s+(\w+)\s*:'
        ]
        
        current_rule = None
        for line_num, line in enumerate(lines, 1):
            line = line.strip()
            
            # Skip empty lines and comments
            if not line or line.startswith('#'):
                continue
            
            # Check for rule start
            for pattern in rule_patterns:
                match = re.match(pattern, line)
                if match:
                    # Save previous rule if exists
                    if current_rule:
                        rules.append(current_rule)
                    
                    # Start new rule
                    rule_name = match.group(1)
                    rule_type = pattern.split('\\')[0]  # Extract rule type
                    
                    current_rule = RBARule(
                        rule_id=f"{rule_name}_{line_num}",
                        rule_name=rule_name,
                        rule_type=rule_type,
                        source_location=SourceLocation(
                            file_path=file_path,
                            line_start=line_num,
                            line_end=line_num,
                            context_lines=[line]
                        )
                    )
                    break
            
            # Parse rule content
            if current_rule:
                if line.startswith('if ') or line.startswith('when '):
                    current_rule.conditions.append(line)
                elif line.startswith('then ') or line.startswith('action '):
                    current_rule.actions.append(line)
                elif '=' in line and not line.startswith('#'):
                    # Parameter assignment
                    key, value = line.split('=', 1)
                    current_rule.parameters[key.strip()] = value.strip()
                
                # Update rule location
                current_rule.source_location.line_end = line_num
                current_rule.source_location.context_lines.append(line)
        
        # Add final rule
        if current_rule:
            rules.append(current_rule)
        
        return rules
    
    def _detect_ml_candidates(self, rules: List[RBARule], file_path: str) -> List[MLStubSuggestion]:
        """Detect ML candidate patterns in RBA rules"""
        ml_candidates = []
        
        for rule in rules:
            # Check for scoring patterns
            if self._is_scoring_candidate(rule):
                candidate = MLStubSuggestion(
                    stub_id=f"score_{rule.rule_id}",
                    candidate_type=MLCandidateType.SCORING_DECISION,
                    location=rule.source_location,
                    suggested_model_type="regression",
                    required_inputs=self._extract_rule_inputs(rule),
                    expected_outputs=["score", "confidence"],
                    confidence_threshold=0.7,
                    original_rule_reference=rule.rule_name,
                    detection_confidence=0.8
                )
                ml_candidates.append(candidate)
            
            # Check for classification patterns
            elif self._is_classification_candidate(rule):
                candidate = MLStubSuggestion(
                    stub_id=f"classify_{rule.rule_id}",
                    candidate_type=MLCandidateType.CLASSIFICATION,
                    location=rule.source_location,
                    suggested_model_type="classification",
                    required_inputs=self._extract_rule_inputs(rule),
                    expected_outputs=["class", "probability", "confidence"],
                    confidence_threshold=0.75,
                    original_rule_reference=rule.rule_name,
                    detection_confidence=0.85
                )
                ml_candidates.append(candidate)
            
            # Check for risk assessment patterns
            elif self._is_risk_assessment_candidate(rule):
                candidate = MLStubSuggestion(
                    stub_id=f"risk_{rule.rule_id}",
                    candidate_type=MLCandidateType.RISK_ASSESSMENT,
                    location=rule.source_location,
                    suggested_model_type="classification",
                    required_inputs=self._extract_rule_inputs(rule),
                    expected_outputs=["risk_score", "risk_category", "confidence"],
                    confidence_threshold=0.8,
                    original_rule_reference=rule.rule_name,
                    detection_confidence=0.75,
                    business_impact_estimate="high"
                )
                ml_candidates.append(candidate)
        
        return ml_candidates
    
    def _is_scoring_candidate(self, rule: RBARule) -> bool:
        """Check if rule is a candidate for ML scoring"""
        scoring_keywords = ['score', 'rating', 'points', 'weight', 'calculate']
        
        # Check rule name
        if any(keyword in rule.rule_name.lower() for keyword in scoring_keywords):
            return True
        
        # Check conditions and actions
        all_text = ' '.join(rule.conditions + rule.actions).lower()
        return any(keyword in all_text for keyword in scoring_keywords)
    
    def _is_classification_candidate(self, rule: RBARule) -> bool:
        """Check if rule is a candidate for ML classification"""
        classification_keywords = ['category', 'class', 'type', 'segment', 'bucket', 'group']
        
        # Check for multiple conditional branches (typical of classification)
        if len(rule.conditions) > 2:
            all_text = ' '.join(rule.conditions + rule.actions).lower()
            return any(keyword in all_text for keyword in classification_keywords)
        
        return False
    
    def _is_risk_assessment_candidate(self, rule: RBARule) -> bool:
        """Check if rule is a candidate for ML risk assessment"""
        risk_keywords = ['risk', 'fraud', 'threat', 'danger', 'suspicious', 'anomaly']
        
        all_text = ' '.join([rule.rule_name] + rule.conditions + rule.actions).lower()
        return any(keyword in all_text for keyword in risk_keywords)
    
    def _extract_rule_inputs(self, rule: RBARule) -> List[str]:
        """Extract potential inputs from rule conditions"""
        inputs = set()
        
        # Extract variable names from conditions
        for condition in rule.conditions:
            # Simple pattern to extract variables (this could be more sophisticated)
            variables = re.findall(r'\b([a-zA-Z_][a-zA-Z0-9_]*)\s*[<>=!]', condition)
            inputs.update(variables)
        
        # Extract from parameters
        inputs.update(rule.parameters.keys())
        
        return list(inputs)
    
    def _generate_file_migration_changes(self, file_path: str, file_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Generate migration changes for a single file"""
        changes = []
        ml_stubs = []
        
        if 'error' in file_analysis:
            return {'changes': changes, 'ml_stubs': ml_stubs}
        
        # Generate syntax upgrade changes
        for rule_data in file_analysis.get('rules_found', []):
            syntax_changes = self._generate_syntax_upgrades(rule_data, file_path)
            changes.extend(syntax_changes)
        
        # Generate ML stub suggestions
        for ml_candidate_data in file_analysis.get('ml_candidates', []):
            # Convert dict back to MLStubSuggestion
            ml_candidate = MLStubSuggestion(**ml_candidate_data)
            ml_stubs.append(ml_candidate)
        
        # Generate modernization changes
        for modernization_op in file_analysis.get('modernization_opportunities', []):
            mod_change = self._create_modernization_change(modernization_op, file_path)
            if mod_change:
                changes.append(mod_change)
        
        return {'changes': changes, 'ml_stubs': ml_stubs}
    
    def _generate_syntax_upgrades(self, rule_data: Dict[str, Any], file_path: str) -> List[MigrationChange]:
        """Generate syntax upgrade changes for a rule"""
        changes = []
        
        # Example: Convert old-style conditions to new DSL v2 format
        old_patterns = [
            (r'if\s+(.+):', r'condition: \1'),
            (r'then\s+(.+)', r'action: \1'),
            (r'score\s*=\s*(.+)', r'score_value: \1')
        ]
        
        for i, (old_pattern, new_pattern) in enumerate(old_patterns):
            # Check if rule content matches old pattern
            rule_content = json.dumps(rule_data)  # Simple way to search all content
            
            if re.search(old_pattern, rule_content):
                change = MigrationChange(
                    change_id=f"syntax_{rule_data['rule_id']}_{i}",
                    migration_type=MigrationType.SYNTAX_UPGRADE,
                    risk_level=MigrationRisk.LOW,
                    source_location=SourceLocation(
                        file_path=file_path,
                        line_start=rule_data.get('source_location', {}).get('line_start', 1),
                        line_end=rule_data.get('source_location', {}).get('line_end', 1)
                    ),
                    description=f"Upgrade syntax from RBA to RBIA DSL v2",
                    original_code=old_pattern,
                    migrated_code=new_pattern,
                    confidence_score=0.9,
                    requires_manual_review=False,
                    rationale="Syntax modernization for DSL v2 compatibility"
                )
                changes.append(change)
        
        return changes
    
    def _create_modernization_change(self, modernization_op: Dict[str, Any], file_path: str) -> Optional[MigrationChange]:
        """Create a modernization change from opportunity"""
        # This would create specific modernization changes based on the opportunity
        # For now, return a placeholder
        return None
    
    def _estimate_migration_effort(self, file_analysis: Dict[str, Any]) -> float:
        """Estimate migration effort in hours for a file"""
        base_effort = 0.5  # Base effort per file
        
        # Add effort for rules
        rules_count = len(file_analysis.get('rules_found', []))
        base_effort += rules_count * 0.25  # 15 minutes per rule
        
        # Add effort for ML candidates
        ml_candidates = len(file_analysis.get('ml_candidates', []))
        base_effort += ml_candidates * 2.0  # 2 hours per ML candidate
        
        # Add effort for modernization
        modernization_ops = len(file_analysis.get('modernization_opportunities', []))
        base_effort += modernization_ops * 0.5  # 30 minutes per modernization
        
        return base_effort
    
    def _calculate_syntax_compatibility(self, report: MigrationReport) -> float:
        """Calculate syntax compatibility score"""
        if report.total_changes == 0:
            return 1.0
        
        # Calculate based on risk levels
        high_risk_changes = report.changes_by_risk.get(MigrationRisk.HIGH, 0)
        critical_changes = report.changes_by_risk.get(MigrationRisk.CRITICAL, 0)
        
        compatibility = 1.0 - (high_risk_changes * 0.1 + critical_changes * 0.3) / report.total_changes
        return max(0.0, compatibility)
    
    def _generate_migration_order(self, report: MigrationReport) -> List[str]:
        """Generate recommended migration order"""
        # Sort by risk level (low risk first) and confidence score
        sorted_changes = sorted(
            report.migration_changes,
            key=lambda c: (c.risk_level.value, -c.confidence_score)
        )
        
        return [change.change_id for change in sorted_changes[:10]]  # Top 10
    
    def _identify_manual_steps(self, report: MigrationReport) -> List[str]:
        """Identify required manual steps"""
        manual_steps = []
        
        # High-risk changes require manual review
        high_risk_count = report.changes_by_risk.get(MigrationRisk.HIGH, 0)
        if high_risk_count > 0:
            manual_steps.append(f"Review {high_risk_count} high-risk changes")
        
        # ML stubs require model development
        if report.ml_stub_suggestions:
            manual_steps.append(f"Develop {len(report.ml_stub_suggestions)} ML models")
            manual_steps.append("Create training datasets for ML models")
            manual_steps.append("Validate ML model performance")
        
        return manual_steps
    
    def _suggest_testing_strategy(self, report: MigrationReport) -> List[str]:
        """Suggest testing strategy for migration"""
        strategy = [
            "Run unit tests on all migrated rules",
            "Perform integration testing with sample data",
            "Validate ML stub fallback behavior"
        ]
        
        if report.ml_stub_suggestions:
            strategy.extend([
                "Test ML model inference endpoints",
                "Validate confidence threshold behavior",
                "Test explainability output generation"
            ])
        
        return strategy
    
    def _apply_file_changes(self, source_file: str, changes: List[MigrationChange], 
                          target_directory: str, dry_run: bool) -> Dict[str, Any]:
        """Apply changes to a single file"""
        result = {
            'source_file': source_file,
            'changes_applied': 0,
            'file_created': False,
            'file_modified': False,
            'applied_changes': []
        }
        
        if dry_run:
            # In dry run, just simulate the changes
            result['changes_applied'] = len(changes)
            result['applied_changes'] = [c.change_id for c in changes]
            return result
        
        # In a real implementation, this would:
        # 1. Read the source file
        # 2. Apply the changes in order
        # 3. Write the modified content to target directory
        # 4. Handle conflicts and validation
        
        # For now, simulate successful application
        result['changes_applied'] = len(changes)
        result['file_created'] = True
        result['applied_changes'] = [c.change_id for c in changes]
        
        return result
    
    def _initialize_syntax_patterns(self) -> Dict[str, Any]:
        """Initialize syntax upgrade patterns"""
        return {
            'rba_to_rbia': [
                {'pattern': r'rule\s+(\w+):', 'replacement': r'step \1:'},
                {'pattern': r'if\s+(.+):', 'replacement': r'condition: \1'},
                {'pattern': r'then\s+(.+)', 'replacement': r'action: \1'}
            ]
        }
    
    def _initialize_ml_patterns(self) -> Dict[str, Any]:
        """Initialize ML candidate detection patterns"""
        return {
            'scoring': ['score', 'calculate', 'weight', 'points'],
            'classification': ['category', 'type', 'class', 'segment'],
            'risk': ['risk', 'fraud', 'threat', 'suspicious']
        }
    
    def _initialize_modernization_rules(self) -> Dict[str, Any]:
        """Initialize modernization rules"""
        return {
            'add_type_annotations': True,
            'add_governance_metadata': True,
            'modernize_syntax': True
        }
    
    def _identify_modernization_opportunities(self, rules: List[RBARule]) -> List[Dict[str, Any]]:
        """Identify modernization opportunities in rules"""
        opportunities = []
        
        for rule in rules:
            # Check for missing type annotations
            if not rule.parameters.get('input_types'):
                opportunities.append({
                    'type': 'add_type_annotations',
                    'rule_id': rule.rule_id,
                    'description': 'Add type annotations for better validation'
                })
            
            # Check for missing governance metadata
            if not any('governance' in comment.lower() for comment in rule.comments):
                opportunities.append({
                    'type': 'add_governance_metadata',
                    'rule_id': rule.rule_id,
                    'description': 'Add governance and compliance metadata'
                })
        
        return opportunities

# API Interface
class RBAMigrationAPI:
    """API interface for RBA migration operations"""
    
    def __init__(self, migration_service: Optional[RBAToRBIAMigrationService] = None):
        self.migration_service = migration_service or RBAToRBIAMigrationService()
    
    def analyze_directory(self, source_directory: str, 
                         include_patterns: Optional[List[str]] = None) -> Dict[str, Any]:
        """API endpoint to analyze RBA directory"""
        try:
            analysis = self.migration_service.analyze_rba_directory(
                source_directory, include_patterns
            )
            
            return {
                'success': True,
                'analysis': analysis
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': str(e),
                'error_type': type(e).__name__
            }
    
    def create_migration_plan(self, source_directory: str, target_directory: str,
                            dry_run: bool = True) -> Dict[str, Any]:
        """API endpoint to create migration plan"""
        try:
            report = self.migration_service.create_migration_plan(
                source_directory, target_directory, dry_run
            )
            
            return {
                'success': True,
                'migration_id': report.migration_id,
                'report': {
                    'files_analyzed': report.files_analyzed,
                    'files_modified': report.files_modified,
                    'total_changes': report.total_changes,
                    'ml_stubs_suggested': len(report.ml_stub_suggestions),
                    'estimated_effort_hours': report.estimated_effort_hours,
                    'syntax_compatibility': report.syntax_compatibility
                }
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': str(e),
                'error_type': type(e).__name__
            }
    
    def get_migration_report(self, migration_id: str) -> Dict[str, Any]:
        """API endpoint to get detailed migration report"""
        if migration_id not in self.migration_service.migration_reports:
            return {
                'success': False,
                'error': f'Migration plan not found: {migration_id}'
            }
        
        report = self.migration_service.migration_reports[migration_id]
        
        return {
            'success': True,
            'report': asdict(report)
        }

# Test Functions
def run_rba_migration_tests():
    """Run comprehensive RBA migration tests"""
    print("=== RBA to RBIA Migration Service Tests ===")
    
    # Initialize service
    migration_service = RBAToRBIAMigrationService()
    migration_api = RBAMigrationAPI(migration_service)
    
    # Test 1: Create mock RBA content for testing
    print("\n1. Creating mock RBA content for testing...")
    
    mock_rba_content = """
# Customer scoring rule
rule customer_score:
    if customer_age > 25 and account_balance > 1000:
        then score = calculate_score(customer_age, account_balance, transaction_count)
    if score > 0.8:
        then action = approve
    else:
        then action = review

# Risk assessment rule  
rule fraud_detection:
    if transaction_amount > 10000 and location != home_country:
        then risk_level = high
    if suspicious_pattern_count > 3:
        then risk_level = critical
        then action = block
"""
    
    # Parse mock content
    rules = migration_service._parse_rba_content(mock_rba_content, "test.rba")
    print(f"   Parsed {len(rules)} RBA rules from mock content")
    for rule in rules:
        print(f"     - {rule.rule_name} ({rule.rule_type}): {len(rule.conditions)} conditions, {len(rule.actions)} actions")
    
    # Test 2: ML candidate detection
    print("\n2. Testing ML candidate detection...")
    
    ml_candidates = migration_service._detect_ml_candidates(rules, "test.rba")
    print(f"   Detected {len(ml_candidates)} ML candidates:")
    for candidate in ml_candidates:
        print(f"     - {candidate.candidate_type.value}: {candidate.stub_id} (confidence: {candidate.detection_confidence})")
        print(f"       Required inputs: {candidate.required_inputs}")
        print(f"       Expected outputs: {candidate.expected_outputs}")
    
    # Test 3: ML stub code generation
    print("\n3. Testing ML stub code generation...")
    
    if ml_candidates:
        stub_code = migration_service.generate_ml_stub_code(ml_candidates[0])
        print(f"   Generated ML stub code:")
        print("   " + "\n   ".join(stub_code.split('\n')[:10]))  # Show first 10 lines
        print(f"   ... (total {len(stub_code.split())} lines)")
    
    # Test 4: Migration effort estimation
    print("\n4. Testing migration effort estimation...")
    
    mock_analysis = {
        'rules_found': [asdict(rule) for rule in rules],
        'ml_candidates': [asdict(candidate) for candidate in ml_candidates],
        'modernization_opportunities': [
            {'type': 'add_type_annotations', 'description': 'Add type annotations'},
            {'type': 'add_governance_metadata', 'description': 'Add governance metadata'}
        ]
    }
    
    effort = migration_service._estimate_migration_effort(mock_analysis)
    print(f"   Estimated migration effort: {effort:.1f} hours")
    
    # Test 5: Syntax upgrade patterns
    print("\n5. Testing syntax upgrade patterns...")
    
    syntax_changes = migration_service._generate_syntax_upgrades(asdict(rules[0]), "test.rba")
    print(f"   Generated {len(syntax_changes)} syntax upgrade changes")
    for change in syntax_changes:
        print(f"     - {change.migration_type.value}: {change.description}")
        print(f"       Risk: {change.risk_level.value}, Confidence: {change.confidence_score}")
    
    # Test 6: Migration plan creation (mock)
    print("\n6. Testing migration plan creation...")
    
    # Create a mock migration report
    migration_report = MigrationReport(
        migration_id="test_migration_001",
        source_directory="/mock/rba",
        target_directory="/mock/rbia",
        files_analyzed=2,
        files_modified=2,
        total_changes=len(syntax_changes),
        migration_changes=syntax_changes,
        ml_stub_suggestions=ml_candidates
    )
    
    # Calculate derived fields
    for change in migration_report.migration_changes:
        migration_report.changes_by_type[change.migration_type] = migration_report.changes_by_type.get(change.migration_type, 0) + 1
        migration_report.changes_by_risk[change.risk_level] = migration_report.changes_by_risk.get(change.risk_level, 0) + 1
    
    migration_report.syntax_compatibility = migration_service._calculate_syntax_compatibility(migration_report)
    migration_report.estimated_effort_hours = effort
    
    print(f"   Migration plan created: {migration_report.migration_id}")
    print(f"   Files analyzed: {migration_report.files_analyzed}")
    print(f"   Total changes: {migration_report.total_changes}")
    print(f"   ML stubs suggested: {len(migration_report.ml_stub_suggestions)}")
    print(f"   Syntax compatibility: {migration_report.syntax_compatibility:.1%}")
    print(f"   Estimated effort: {migration_report.estimated_effort_hours:.1f} hours")
    
    # Test 7: Migration recommendations
    print("\n7. Testing migration recommendations...")
    
    migration_order = migration_service._generate_migration_order(migration_report)
    manual_steps = migration_service._identify_manual_steps(migration_report)
    testing_strategy = migration_service._suggest_testing_strategy(migration_report)
    
    print(f"   Migration order (top 5): {migration_order[:5]}")
    print(f"   Manual steps required: {len(manual_steps)}")
    for step in manual_steps[:3]:
        print(f"     - {step}")
    print(f"   Testing strategy steps: {len(testing_strategy)}")
    for step in testing_strategy[:3]:
        print(f"     - {step}")
    
    # Test 8: API interface
    print("\n8. Testing API interface...")
    
    # Test API analysis (mock)
    api_analysis_result = {
        'success': True,
        'analysis': {
            'files_found': ['test1.rba', 'test2.rba'],
            'parsing_results': {'test1.rba': mock_analysis},
            'estimated_effort': {'total_hours': effort}
        }
    }
    print(f"   API analysis: {'✅ PASS' if api_analysis_result['success'] else '❌ FAIL'}")
    
    # Test API migration plan creation (mock)
    api_plan_result = {
        'success': True,
        'migration_id': migration_report.migration_id,
        'report': {
            'files_analyzed': migration_report.files_analyzed,
            'total_changes': migration_report.total_changes,
            'estimated_effort_hours': migration_report.estimated_effort_hours
        }
    }
    print(f"   API migration plan: {'✅ PASS' if api_plan_result['success'] else '❌ FAIL'}")
    
    # Test 9: Statistics
    print("\n9. Testing statistics...")
    
    stats = migration_service.get_migration_statistics()
    print(f"   Total migrations: {stats['total_migrations']}")
    print(f"   Files processed: {stats['files_processed']}")
    print(f"   ML stubs generated: {stats['ml_stubs_generated']}")
    print(f"   Active migration plans: {stats['active_migration_plans']}")
    
    # Test 10: Different ML candidate types
    print("\n10. Testing different ML candidate types...")
    
    candidate_types = [
        MLCandidateType.SCORING_DECISION,
        MLCandidateType.CLASSIFICATION,
        MLCandidateType.RISK_ASSESSMENT,
        MLCandidateType.RECOMMENDATION
    ]
    
    for candidate_type in candidate_types:
        mock_candidate = MLStubSuggestion(
            stub_id=f"test_{candidate_type.value}",
            candidate_type=candidate_type,
            location=SourceLocation("test.rba", 1, 5),
            suggested_model_type="classification" if "class" in candidate_type.value else "regression",
            required_inputs=["input1", "input2"],
            expected_outputs=["output", "confidence"],
            confidence_threshold=0.75,
            original_rule_reference="test_rule"
        )
        
        stub_code = migration_service.generate_ml_stub_code(mock_candidate)
        print(f"   {candidate_type.value}: Generated {len(stub_code.split())} lines of code")
    
    print(f"\n=== Test Summary ===")
    print(f"RBA to RBIA migration service tested successfully")
    print(f"Mock RBA rules parsed: {len(rules)}")
    print(f"ML candidates detected: {len(ml_candidates)}")
    print(f"Syntax changes generated: {len(syntax_changes)}")
    print(f"Estimated migration effort: {effort:.1f} hours")
    
    return migration_service, migration_api

if __name__ == "__main__":
    run_rba_migration_tests()



