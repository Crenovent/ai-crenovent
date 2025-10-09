"""
Auto-Remediation Suggestions - Task 6.2.64
===========================================

Dev speed lint fixer with quick actions
- Automatic fixes for common linting issues
- Quick action suggestions for developers
- Code generation for remediation
"""

from typing import Dict, List, Any, Optional
import logging
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)

class RemediationType(Enum):
    AUTO_FIX = "auto_fix"           # Can be applied automatically
    SUGGESTED_FIX = "suggested_fix"  # Requires developer approval
    MANUAL_FIX = "manual_fix"       # Requires manual intervention

@dataclass
class RemediationSuggestion:
    issue_id: str
    issue_type: str
    remediation_type: RemediationType
    description: str
    fix_code: Optional[str]  # Code to apply the fix
    explanation: str
    confidence: float

class AutoRemediationService:
    """Service for generating automatic remediation suggestions"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.remediation_rules = self._initialize_remediation_rules()
    
    def _initialize_remediation_rules(self) -> Dict[str, Dict[str, Any]]:
        """Initialize remediation rules for common issues"""
        return {
            'missing_confidence_threshold': {
                'remediation_type': RemediationType.AUTO_FIX,
                'fix_template': '''
trust_budget:
  confidence_threshold: 0.8
  min_confidence_auto: 0.9
''',
                'explanation': 'Added default confidence thresholds for ML node'
            },
            'missing_fallback': {
                'remediation_type': RemediationType.SUGGESTED_FIX,
                'fix_template': '''
fallbacks:
  - condition: "confidence < 0.8"
    target: "manual_review"
    reason: "Low confidence prediction"
''',
                'explanation': 'Added fallback for low confidence scenarios'
            },
            'missing_region_id': {
                'remediation_type': RemediationType.AUTO_FIX,
                'fix_template': '''
policies:
  region_id: "us-east-1"
''',
                'explanation': 'Added default region for compliance'
            },
            'insecure_secret_reference': {
                'remediation_type': RemediationType.SUGGESTED_FIX,
                'fix_template': '''
config:
  api_key: "vault://secrets/api_key"
''',
                'explanation': 'Replaced plaintext secret with vault reference'
            },
            'missing_approval_gate': {
                'remediation_type': RemediationType.SUGGESTED_FIX,
                'fix_template': '''
policies:
  approval_required: true
  approver_role: "ml_architect"
''',
                'explanation': 'Added approval gate for ML deployment'
            },
            'missing_bias_monitoring': {
                'remediation_type': RemediationType.AUTO_FIX,
                'fix_template': '''
monitoring:
  bias_detection: true
  drift_detection: true
  monitoring_frequency: "daily"
''',
                'explanation': 'Added bias and drift monitoring configuration'
            },
            'unreachable_code': {
                'remediation_type': RemediationType.MANUAL_FIX,
                'fix_template': None,
                'explanation': 'Remove unreachable code paths or fix conditions'
            },
            'missing_error_handling': {
                'remediation_type': RemediationType.SUGGESTED_FIX,
                'fix_template': '''
error_handling:
  on_failure: "fallback_to_manual"
  retry_count: 3
  timeout_seconds: 30
''',
                'explanation': 'Added error handling configuration'
            }
        }
    
    def analyze_and_suggest_remediations(self, ir_data: Dict[str, Any], 
                                       lint_issues: List[Dict[str, Any]]) -> List[RemediationSuggestion]:
        """Analyze lint issues and generate remediation suggestions"""
        suggestions = []
        
        for issue in lint_issues:
            issue_type = issue.get('type', 'unknown')
            issue_id = issue.get('id', 'unknown')
            node_id = issue.get('node_id')
            
            # Generate remediation based on issue type
            if issue_type in self.remediation_rules:
                suggestion = self._create_remediation_suggestion(issue, ir_data)
                if suggestion:
                    suggestions.append(suggestion)
            else:
                # Generic remediation for unknown issues
                suggestion = self._create_generic_remediation(issue)
                suggestions.append(suggestion)
        
        return suggestions
    
    def _create_remediation_suggestion(self, issue: Dict[str, Any], 
                                     ir_data: Dict[str, Any]) -> Optional[RemediationSuggestion]:
        """Create specific remediation suggestion for known issue types"""
        issue_type = issue.get('type')
        issue_id = issue.get('id', 'unknown')
        node_id = issue.get('node_id')
        
        rule = self.remediation_rules.get(issue_type)
        if not rule:
            return None
        
        # Generate fix code based on context
        fix_code = self._generate_fix_code(issue_type, issue, ir_data, rule)
        
        return RemediationSuggestion(
            issue_id=issue_id,
            issue_type=issue_type,
            remediation_type=rule['remediation_type'],
            description=f"Fix {issue_type.replace('_', ' ')} in {node_id or 'plan'}",
            fix_code=fix_code,
            explanation=rule['explanation'],
            confidence=0.9 if rule['remediation_type'] == RemediationType.AUTO_FIX else 0.7
        )
    
    def _create_generic_remediation(self, issue: Dict[str, Any]) -> RemediationSuggestion:
        """Create generic remediation for unknown issue types"""
        return RemediationSuggestion(
            issue_id=issue.get('id', 'unknown'),
            issue_type=issue.get('type', 'unknown'),
            remediation_type=RemediationType.MANUAL_FIX,
            description=f"Manual fix required for {issue.get('type', 'unknown issue')}",
            fix_code=None,
            explanation="This issue requires manual review and fixing",
            confidence=0.5
        )
    
    def _generate_fix_code(self, issue_type: str, issue: Dict[str, Any], 
                          ir_data: Dict[str, Any], rule: Dict[str, Any]) -> Optional[str]:
        """Generate specific fix code based on issue context"""
        fix_template = rule.get('fix_template')
        if not fix_template:
            return None
        
        node_id = issue.get('node_id')
        
        # Customize fix based on issue type and context
        if issue_type == 'missing_confidence_threshold':
            # Adjust threshold based on node type
            node = self._find_node_by_id(ir_data, node_id)
            if node and 'critical' in node.get('description', '').lower():
                return fix_template.replace('0.8', '0.9').replace('0.9', '0.95')
        
        elif issue_type == 'missing_region_id':
            # Use tenant's default region if available
            tenant_region = ir_data.get('metadata', {}).get('default_region', 'us-east-1')
            return fix_template.replace('us-east-1', tenant_region)
        
        elif issue_type == 'insecure_secret_reference':
            # Generate vault path based on context
            secret_name = issue.get('secret_name', 'api_key')
            vault_path = f"vault://secrets/{secret_name}"
            return fix_template.replace('vault://secrets/api_key', vault_path)
        
        return fix_template
    
    def apply_auto_fixes(self, ir_data: Dict[str, Any], 
                        suggestions: List[RemediationSuggestion]) -> Dict[str, Any]:
        """Apply automatic fixes to the IR"""
        modified_ir = ir_data.copy()
        applied_fixes = []
        
        for suggestion in suggestions:
            if (suggestion.remediation_type == RemediationType.AUTO_FIX and 
                suggestion.fix_code and 
                suggestion.confidence > 0.8):
                
                # Apply the fix
                success = self._apply_fix_to_ir(modified_ir, suggestion)
                if success:
                    applied_fixes.append(suggestion.issue_id)
                    self.logger.info(f"Applied auto-fix for {suggestion.issue_type}")
        
        # Add metadata about applied fixes
        if 'metadata' not in modified_ir:
            modified_ir['metadata'] = {}
        
        modified_ir['metadata']['applied_auto_fixes'] = applied_fixes
        
        return modified_ir
    
    def _apply_fix_to_ir(self, ir_data: Dict[str, Any], 
                        suggestion: RemediationSuggestion) -> bool:
        """Apply a specific fix to the IR data"""
        try:
            import yaml
            
            # Parse the fix code as YAML
            fix_data = yaml.safe_load(suggestion.fix_code)
            
            # Determine where to apply the fix
            if suggestion.issue_type in ['missing_confidence_threshold', 'missing_fallback']:
                # Apply to specific node
                node_id = self._extract_node_id_from_issue(suggestion.issue_id)
                node = self._find_node_by_id(ir_data, node_id)
                if node:
                    self._merge_dict(node, fix_data)
                    return True
            
            elif suggestion.issue_type in ['missing_region_id', 'missing_approval_gate']:
                # Apply to plan policies
                if 'policies' not in ir_data:
                    ir_data['policies'] = {}
                self._merge_dict(ir_data['policies'], fix_data.get('policies', {}))
                return True
            
            elif suggestion.issue_type == 'missing_bias_monitoring':
                # Apply to plan metadata
                if 'metadata' not in ir_data:
                    ir_data['metadata'] = {}
                self._merge_dict(ir_data['metadata'], fix_data)
                return True
            
            return False
            
        except Exception as e:
            self.logger.error(f"Failed to apply fix: {e}")
            return False
    
    def _merge_dict(self, target: Dict[str, Any], source: Dict[str, Any]):
        """Merge source dictionary into target dictionary"""
        for key, value in source.items():
            if key in target and isinstance(target[key], dict) and isinstance(value, dict):
                self._merge_dict(target[key], value)
            else:
                target[key] = value
    
    def _find_node_by_id(self, ir_data: Dict[str, Any], node_id: str) -> Optional[Dict[str, Any]]:
        """Find node by ID in IR data"""
        nodes = ir_data.get('nodes', [])
        for node in nodes:
            if node.get('id') == node_id:
                return node
        return None
    
    def _extract_node_id_from_issue(self, issue_id: str) -> Optional[str]:
        """Extract node ID from issue ID"""
        # Assuming issue ID format like "missing_confidence_threshold_node_123"
        parts = issue_id.split('_')
        if len(parts) >= 2 and parts[-2] == 'node':
            return parts[-1]
        return None
    
    def generate_quick_actions(self, suggestions: List[RemediationSuggestion]) -> List[Dict[str, Any]]:
        """Generate quick actions for IDE/UI integration"""
        quick_actions = []
        
        for suggestion in suggestions:
            action = {
                'id': f"quick_fix_{suggestion.issue_id}",
                'title': f"Fix {suggestion.issue_type.replace('_', ' ').title()}",
                'description': suggestion.explanation,
                'action_type': suggestion.remediation_type.value,
                'confidence': suggestion.confidence,
                'applicable': suggestion.fix_code is not None
            }
            
            if suggestion.remediation_type == RemediationType.AUTO_FIX:
                action['button_text'] = "Apply Fix"
                action['button_style'] = "primary"
            elif suggestion.remediation_type == RemediationType.SUGGESTED_FIX:
                action['button_text'] = "Review & Apply"
                action['button_style'] = "secondary"
            else:
                action['button_text'] = "View Details"
                action['button_style'] = "outline"
            
            quick_actions.append(action)
        
        return quick_actions
    
    def get_remediation_statistics(self, suggestions: List[RemediationSuggestion]) -> Dict[str, Any]:
        """Get statistics about remediation suggestions"""
        total_suggestions = len(suggestions)
        auto_fixable = len([s for s in suggestions if s.remediation_type == RemediationType.AUTO_FIX])
        suggested_fixes = len([s for s in suggestions if s.remediation_type == RemediationType.SUGGESTED_FIX])
        manual_fixes = len([s for s in suggestions if s.remediation_type == RemediationType.MANUAL_FIX])
        
        # Group by issue type
        issue_type_counts = {}
        for suggestion in suggestions:
            issue_type = suggestion.issue_type
            issue_type_counts[issue_type] = issue_type_counts.get(issue_type, 0) + 1
        
        return {
            'total_suggestions': total_suggestions,
            'auto_fixable': auto_fixable,
            'suggested_fixes': suggested_fixes,
            'manual_fixes': manual_fixes,
            'auto_fix_percentage': (auto_fixable / total_suggestions * 100) if total_suggestions > 0 else 0,
            'issue_type_distribution': issue_type_counts
        }
