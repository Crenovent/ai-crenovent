"""
DSL Scaffolder - Task 6.5.75
=============================

Conversational intents → DSL scaffolder
- Fast prototyping from natural language
- Generates DSL templates from intents
- Policy hints included in scaffolding
"""

from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from enum import Enum
import logging

logger = logging.getLogger(__name__)

class IntentCategory(Enum):
    DATA_ANALYSIS = "data_analysis"
    RISK_ASSESSMENT = "risk_assessment"
    APPROVAL_WORKFLOW = "approval_workflow"
    NOTIFICATION = "notification"
    VALIDATION = "validation"

@dataclass
class ScaffoldRequest:
    """Request for DSL scaffolding"""
    intent_text: str
    tenant_id: str
    context: Dict[str, Any]

@dataclass
class ScaffoldResult:
    """Result of DSL scaffolding"""
    dsl_code: str
    intent_category: IntentCategory
    confidence: float
    policy_hints: List[str]

class DSLScaffolder:
    """Scaffolds DSL from conversational intents"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.intent_templates = self._load_intent_templates()
        self.intent_patterns = self._load_intent_patterns()
    
    def scaffold_from_text(self, request: ScaffoldRequest) -> ScaffoldResult:
        """Generate DSL scaffold from natural language"""
        # Classify intent
        category, confidence = self._classify_intent(request.intent_text)
        
        # Generate DSL
        template = self.intent_templates.get(category, self._get_default_template())
        dsl_code = template.format(
            workflow_id=f"generated_{category.value}_workflow",
            intent_text=request.intent_text,
            tenant_id=request.tenant_id,
            region_id=request.context.get('region_id', '{{ region_id }}'),
            policy_pack=request.context.get('policy_pack', 'default_policy')
        )
        
        # Generate policy hints
        policy_hints = self._generate_policy_hints(category, request.context)
        
        return ScaffoldResult(
            dsl_code=dsl_code,
            intent_category=category,
            confidence=confidence,
            policy_hints=policy_hints
        )
    
    def _classify_intent(self, text: str) -> tuple[IntentCategory, float]:
        """Classify intent from text"""
        text_lower = text.lower()
        
        for category, patterns in self.intent_patterns.items():
            for pattern in patterns:
                if pattern in text_lower:
                    return category, 0.85
        
        return IntentCategory.VALIDATION, 0.5
    
    def _load_intent_patterns(self) -> Dict[IntentCategory, List[str]]:
        """Load patterns for intent classification"""
        return {
            IntentCategory.DATA_ANALYSIS: [
                "analyze", "report", "dashboard", "metrics", "performance"
            ],
            IntentCategory.RISK_ASSESSMENT: [
                "risk", "assess", "evaluate", "score", "threat"
            ],
            IntentCategory.APPROVAL_WORKFLOW: [
                "approve", "review", "authorize", "permission", "workflow"
            ],
            IntentCategory.NOTIFICATION: [
                "notify", "alert", "send", "email", "message"
            ],
            IntentCategory.VALIDATION: [
                "validate", "check", "verify", "confirm", "ensure"
            ]
        }
    
    def _load_intent_templates(self) -> Dict[IntentCategory, str]:
        """Load DSL templates for each intent category"""
        return {
            IntentCategory.DATA_ANALYSIS: '''workflow_id: "{workflow_id}"
name: "Data Analysis: {intent_text}"
module: "data_analysis"
automation_type: "RBIA"
version: "1.0.0"
policy_pack: "{policy_pack}"

governance:
  tenant_id: "{tenant_id}"
  region_id: "{region_id}"
  sla_tier: "T1"
  evidence_capture: true

steps:
  - id: "analyze_data"
    type: "ml_node"
    params:
      model_id: "data_analyzer_v1"
      confidence_threshold: 0.8
    fallback:
      enabled: true
      fallback: ["rba_analysis"]
''',
            IntentCategory.RISK_ASSESSMENT: '''workflow_id: "{workflow_id}"
name: "Risk Assessment: {intent_text}"
module: "risk_assessment"
automation_type: "RBIA"
version: "1.0.0"
policy_pack: "{policy_pack}"

governance:
  tenant_id: "{tenant_id}"
  region_id: "{region_id}"
  sla_tier: "T0"
  evidence_capture: true

steps:
  - id: "assess_risk"
    type: "ml_score"
    params:
      model_id: "risk_scorer_v1"
      score_threshold: 70
    fallback:
      enabled: true
      fallback: ["manual_review"]
'''
        }
    
    def _get_default_template(self) -> str:
        """Get default DSL template"""
        return '''workflow_id: "{workflow_id}"
name: "Generated Workflow: {intent_text}"
module: "general"
automation_type: "RBA"
version: "1.0.0"
policy_pack: "{policy_pack}"

governance:
  tenant_id: "{tenant_id}"
  region_id: "{region_id}"
  sla_tier: "T2"

steps:
  - id: "execute_action"
    type: "decision"
    params:
      action: "process_request"
'''
    
    def _generate_policy_hints(self, category: IntentCategory, context: Dict[str, Any]) -> List[str]:
        """Generate policy hints for the category"""
        hints = []
        
        if category == IntentCategory.RISK_ASSESSMENT:
            hints.extend([
                "Consider adding SOX compliance policy for financial risk",
                "Add evidence_pack_required: true for audit trail",
                "Set confidence_threshold >= 0.8 for risk decisions"
            ])
        elif category == IntentCategory.DATA_ANALYSIS:
            hints.extend([
                "Add GDPR policy if processing personal data",
                "Consider data residency constraints",
                "Set appropriate SLA tier based on criticality"
            ])
        elif category == IntentCategory.APPROVAL_WORKFLOW:
            hints.extend([
                "Add SoD (Separation of Duties) validation",
                "Include approver role validation",
                "Set evidence capture for compliance"
            ])
        
        return hints
