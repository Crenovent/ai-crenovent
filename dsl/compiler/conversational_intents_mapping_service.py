"""
Conversational Intents Mapping - Task 6.2.59
=============================================

NLU bridge mapping conversational intents to plan stubs
- Maps natural language intents to workflow templates
- For 3.3 flows integration
- Intent registry for plan generation
"""

from typing import Dict, List, Any, Optional
import logging
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)

class IntentCategory(Enum):
    DATA_ANALYSIS = "data_analysis"
    RISK_ASSESSMENT = "risk_assessment"
    APPROVAL_WORKFLOW = "approval_workflow"
    NOTIFICATION = "notification"
    VALIDATION = "validation"

@dataclass
class ConversationalIntent:
    intent_id: str
    intent_name: str
    category: IntentCategory
    example_phrases: List[str]
    plan_template_id: str
    required_parameters: List[str]
    optional_parameters: List[str]

@dataclass
class PlanStub:
    plan_id: str
    template_id: str
    parameters: Dict[str, Any]
    nodes: List[Dict[str, Any]]
    edges: List[Dict[str, Any]]

class ConversationalIntentsMappingService:
    """Service for mapping conversational intents to plan stubs"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.intent_registry = self._initialize_intent_registry()
    
    def _initialize_intent_registry(self) -> Dict[str, ConversationalIntent]:
        """Initialize intent registry with common intents"""
        intents = [
            ConversationalIntent(
                intent_id="analyze_pipeline_health",
                intent_name="Analyze Pipeline Health",
                category=IntentCategory.DATA_ANALYSIS,
                example_phrases=[
                    "check pipeline health",
                    "analyze my pipeline",
                    "how is my pipeline performing"
                ],
                plan_template_id="pipeline_health_analysis",
                required_parameters=["tenant_id"],
                optional_parameters=["date_range", "pipeline_id"]
            ),
            ConversationalIntent(
                intent_id="assess_deal_risk",
                intent_name="Assess Deal Risk",
                category=IntentCategory.RISK_ASSESSMENT,
                example_phrases=[
                    "check deal risk",
                    "assess this opportunity",
                    "what's the risk level"
                ],
                plan_template_id="deal_risk_assessment",
                required_parameters=["tenant_id", "deal_id"],
                optional_parameters=["risk_factors"]
            ),
            ConversationalIntent(
                intent_id="request_approval",
                intent_name="Request Approval",
                category=IntentCategory.APPROVAL_WORKFLOW,
                example_phrases=[
                    "request approval for",
                    "need approval",
                    "submit for review"
                ],
                plan_template_id="approval_workflow",
                required_parameters=["tenant_id", "item_id", "approver"],
                optional_parameters=["urgency", "justification"]
            )
        ]
        
        return {intent.intent_id: intent for intent in intents}
    
    def match_intent(self, user_input: str) -> Optional[ConversationalIntent]:
        """Match user input to registered intent"""
        user_input_lower = user_input.lower()
        
        for intent in self.intent_registry.values():
            for phrase in intent.example_phrases:
                if phrase.lower() in user_input_lower:
                    return intent
        
        return None
    
    def generate_plan_stub(self, intent: ConversationalIntent, 
                          parameters: Dict[str, Any]) -> PlanStub:
        """Generate plan stub from intent and parameters"""
        
        # Validate required parameters
        missing_params = [p for p in intent.required_parameters if p not in parameters]
        if missing_params:
            raise ValueError(f"Missing required parameters: {missing_params}")
        
        plan_id = f"plan_{intent.intent_id}_{parameters.get('tenant_id', 'unknown')}"
        
        if intent.intent_id == "analyze_pipeline_health":
            return self._create_pipeline_health_stub(plan_id, intent.plan_template_id, parameters)
        elif intent.intent_id == "assess_deal_risk":
            return self._create_deal_risk_stub(plan_id, intent.plan_template_id, parameters)
        elif intent.intent_id == "request_approval":
            return self._create_approval_stub(plan_id, intent.plan_template_id, parameters)
        else:
            return self._create_generic_stub(plan_id, intent.plan_template_id, parameters)
    
    def _create_pipeline_health_stub(self, plan_id: str, template_id: str, 
                                   parameters: Dict[str, Any]) -> PlanStub:
        """Create pipeline health analysis plan stub"""
        return PlanStub(
            plan_id=plan_id,
            template_id=template_id,
            parameters=parameters,
            nodes=[
                {
                    "id": "data_fetch",
                    "type": "data_source",
                    "name": "Fetch Pipeline Data",
                    "config": {"tenant_id": parameters["tenant_id"]}
                },
                {
                    "id": "health_analysis",
                    "type": "ml_node",
                    "name": "Pipeline Health Analysis",
                    "config": {"model": "pipeline_health_classifier"}
                },
                {
                    "id": "generate_report",
                    "type": "report",
                    "name": "Generate Health Report"
                }
            ],
            edges=[
                {"source": "data_fetch", "target": "health_analysis"},
                {"source": "health_analysis", "target": "generate_report"}
            ]
        )
    
    def _create_deal_risk_stub(self, plan_id: str, template_id: str, 
                             parameters: Dict[str, Any]) -> PlanStub:
        """Create deal risk assessment plan stub"""
        return PlanStub(
            plan_id=plan_id,
            template_id=template_id,
            parameters=parameters,
            nodes=[
                {
                    "id": "fetch_deal",
                    "type": "data_source",
                    "name": "Fetch Deal Data",
                    "config": {
                        "tenant_id": parameters["tenant_id"],
                        "deal_id": parameters["deal_id"]
                    }
                },
                {
                    "id": "risk_assessment",
                    "type": "ml_node",
                    "name": "Risk Assessment ML",
                    "config": {"model": "deal_risk_scorer"}
                },
                {
                    "id": "risk_decision",
                    "type": "decision",
                    "name": "Risk Decision Gate"
                }
            ],
            edges=[
                {"source": "fetch_deal", "target": "risk_assessment"},
                {"source": "risk_assessment", "target": "risk_decision"}
            ]
        )
    
    def _create_approval_stub(self, plan_id: str, template_id: str, 
                            parameters: Dict[str, Any]) -> PlanStub:
        """Create approval workflow plan stub"""
        return PlanStub(
            plan_id=plan_id,
            template_id=template_id,
            parameters=parameters,
            nodes=[
                {
                    "id": "prepare_request",
                    "type": "data_transform",
                    "name": "Prepare Approval Request"
                },
                {
                    "id": "send_notification",
                    "type": "notification",
                    "name": "Notify Approver",
                    "config": {"approver": parameters["approver"]}
                },
                {
                    "id": "await_approval",
                    "type": "human_task",
                    "name": "Await Approval Decision"
                }
            ],
            edges=[
                {"source": "prepare_request", "target": "send_notification"},
                {"source": "send_notification", "target": "await_approval"}
            ]
        )
    
    def _create_generic_stub(self, plan_id: str, template_id: str, 
                           parameters: Dict[str, Any]) -> PlanStub:
        """Create generic plan stub"""
        return PlanStub(
            plan_id=plan_id,
            template_id=template_id,
            parameters=parameters,
            nodes=[
                {
                    "id": "start",
                    "type": "start",
                    "name": "Start"
                },
                {
                    "id": "process",
                    "type": "step",
                    "name": "Process"
                },
                {
                    "id": "end",
                    "type": "end",
                    "name": "End"
                }
            ],
            edges=[
                {"source": "start", "target": "process"},
                {"source": "process", "target": "end"}
            ]
        )
    
    def get_intent_suggestions(self, partial_input: str) -> List[ConversationalIntent]:
        """Get intent suggestions based on partial input"""
        suggestions = []
        partial_lower = partial_input.lower()
        
        for intent in self.intent_registry.values():
            for phrase in intent.example_phrases:
                if partial_lower in phrase.lower() or phrase.lower().startswith(partial_lower):
                    suggestions.append(intent)
                    break
        
        return suggestions
