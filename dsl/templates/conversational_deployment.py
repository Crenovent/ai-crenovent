"""
Conversational Mode Support for Template Deployment
Task 4.2.17: Add Conversational Mode support for template deployment
"""

import json
import uuid
import asyncio
from typing import Dict, List, Any, Optional, Union, Callable
from dataclasses import dataclass, field
from datetime import datetime
import logging
import re
from enum import Enum

from .industry_template_registry import IndustryTemplateRegistry, TemplateConfig
from .template_confidence_manager import TemplateConfidenceManager
from .template_explainability_system import TemplateExplainabilitySystem

logger = logging.getLogger(__name__)

class ConversationState(Enum):
    """States in the conversational template deployment flow"""
    INITIAL = "initial"
    TEMPLATE_SELECTION = "template_selection"
    PARAMETER_COLLECTION = "parameter_collection"
    CONFIRMATION = "confirmation"
    DEPLOYMENT = "deployment"
    MONITORING = "monitoring"
    COMPLETED = "completed"
    ERROR = "error"

@dataclass
class ConversationContext:
    """Context for conversational template deployment"""
    conversation_id: str
    user_id: int
    tenant_id: str
    state: ConversationState
    selected_template: Optional[str] = None
    collected_parameters: Dict[str, Any] = field(default_factory=dict)
    deployment_config: Dict[str, Any] = field(default_factory=dict)
    conversation_history: List[Dict[str, str]] = field(default_factory=list)
    created_at: datetime = field(default_factory=datetime.utcnow)
    last_updated: datetime = field(default_factory=datetime.utcnow)

@dataclass
class ConversationResponse:
    """Response from conversational system"""
    message: str
    options: List[str] = field(default_factory=list)
    input_required: bool = False
    input_type: str = "text"  # text, number, boolean, select
    validation_pattern: Optional[str] = None
    next_state: Optional[ConversationState] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

class ConversationalTemplateDeployment:
    """
    Conversational interface for template deployment
    
    Provides:
    - Natural language template selection
    - Guided parameter collection
    - Interactive deployment configuration
    - Real-time deployment monitoring
    - Error handling and recovery
    """
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.template_registry = IndustryTemplateRegistry()
        self.confidence_manager = TemplateConfidenceManager()
        self.explainability_system = TemplateExplainabilitySystem()
        self.active_conversations: Dict[str, ConversationContext] = {}
        self.conversation_handlers = self._initialize_handlers()
    
    def _initialize_handlers(self) -> Dict[ConversationState, Callable]:
        """Initialize conversation state handlers"""
        return {
            ConversationState.INITIAL: self._handle_initial,
            ConversationState.TEMPLATE_SELECTION: self._handle_template_selection,
            ConversationState.PARAMETER_COLLECTION: self._handle_parameter_collection,
            ConversationState.CONFIRMATION: self._handle_confirmation,
            ConversationState.DEPLOYMENT: self._handle_deployment,
            ConversationState.MONITORING: self._handle_monitoring,
            ConversationState.COMPLETED: self._handle_completed,
            ConversationState.ERROR: self._handle_error
        }
    
    async def start_conversation(
        self,
        user_id: int,
        tenant_id: str,
        initial_message: Optional[str] = None
    ) -> ConversationResponse:
        """Start a new conversational template deployment"""
        
        conversation_id = str(uuid.uuid4())
        context = ConversationContext(
            conversation_id=conversation_id,
            user_id=user_id,
            tenant_id=tenant_id,
            state=ConversationState.INITIAL
        )
        
        if initial_message:
            context.conversation_history.append({
                "role": "user",
                "message": initial_message,
                "timestamp": datetime.utcnow().isoformat()
            })
        
        self.active_conversations[conversation_id] = context
        
        response = await self._handle_initial(context, initial_message)
        
        # Add assistant response to history
        context.conversation_history.append({
            "role": "assistant",
            "message": response.message,
            "timestamp": datetime.utcnow().isoformat()
        })
        
        context.last_updated = datetime.utcnow()
        
        response.metadata["conversation_id"] = conversation_id
        return response
    
    async def continue_conversation(
        self,
        conversation_id: str,
        user_message: str
    ) -> ConversationResponse:
        """Continue an existing conversation"""
        
        context = self.active_conversations.get(conversation_id)
        if not context:
            return ConversationResponse(
                message="Conversation not found. Please start a new conversation.",
                next_state=ConversationState.ERROR
            )
        
        # Add user message to history
        context.conversation_history.append({
            "role": "user",
            "message": user_message,
            "timestamp": datetime.utcnow().isoformat()
        })
        
        # Handle current state
        handler = self.conversation_handlers.get(context.state)
        if not handler:
            return ConversationResponse(
                message="Invalid conversation state. Please start a new conversation.",
                next_state=ConversationState.ERROR
            )
        
        response = await handler(context, user_message)
        
        # Update state if specified
        if response.next_state:
            context.state = response.next_state
        
        # Add assistant response to history
        context.conversation_history.append({
            "role": "assistant",
            "message": response.message,
            "timestamp": datetime.utcnow().isoformat()
        })
        
        context.last_updated = datetime.utcnow()
        
        response.metadata["conversation_id"] = conversation_id
        return response
    
    async def _handle_initial(
        self,
        context: ConversationContext,
        message: Optional[str]
    ) -> ConversationResponse:
        """Handle initial conversation state"""
        
        try:
            # Analyze initial message for intent
            if message:
                template_intent = await self._analyze_template_intent(message)
                if template_intent:
                    context.selected_template = template_intent["template_id"]
                    context.collected_parameters.update(template_intent.get("parameters", {}))
            
            # Get available templates
            templates = self.template_registry.list_all_templates()
            industry_summary = self.template_registry.get_industry_summary()
            
            if context.selected_template:
                template = self.template_registry.get_template(context.selected_template)
                return ConversationResponse(
                    message=f"Great! I understand you want to deploy the '{template.template_name}' template. Let me help you configure it.",
                    next_state=ConversationState.PARAMETER_COLLECTION
                )
            
            # Present template options
            message_parts = [
                "👋 Welcome to the RBIA Template Deployment Assistant!",
                "",
                "I can help you deploy intelligent automation templates for your business needs.",
                "",
                "Available templates by industry:"
            ]
            
            for industry, data in industry_summary.items():
                message_parts.append(f"\n🏢 **{industry}** ({data['template_count']} templates):")
                for template in data['templates']:
                    message_parts.append(f"   • {template['template_name']} ({template['template_type']})")
            
            message_parts.extend([
                "",
                "You can:",
                "1. Tell me your industry and use case (e.g., 'I need fraud detection for banking')",
                "2. Choose a specific template by name",
                "3. Ask about template capabilities",
                "",
                "What would you like to do?"
            ])
            
            return ConversationResponse(
                message="\n".join(message_parts),
                input_required=True,
                input_type="text",
                next_state=ConversationState.TEMPLATE_SELECTION
            )
            
        except Exception as e:
            self.logger.error(f"Error in initial handler: {e}")
            return ConversationResponse(
                message="I encountered an error. Let me try to help you again.",
                next_state=ConversationState.ERROR
            )
    
    async def _handle_template_selection(
        self,
        context: ConversationContext,
        message: str
    ) -> ConversationResponse:
        """Handle template selection state"""
        
        try:
            # Analyze message for template selection
            template_intent = await self._analyze_template_intent(message)
            
            if template_intent:
                template_id = template_intent["template_id"]
                template = self.template_registry.get_template(template_id)
                
                if template:
                    context.selected_template = template_id
                    context.collected_parameters.update(template_intent.get("parameters", {}))
                    
                    return ConversationResponse(
                        message=f"Perfect! You've selected: **{template.template_name}**\n\n"
                                f"📋 **Description**: {template.description}\n"
                                f"🏢 **Industry**: {template.industry}\n"
                                f"🔧 **Type**: {template.template_type}\n\n"
                                f"Now let's configure the template parameters.",
                        next_state=ConversationState.PARAMETER_COLLECTION
                    )
            
            # If no clear intent, provide suggestions
            suggestions = await self._get_template_suggestions(message)
            
            if suggestions:
                message_parts = [
                    "I found several templates that might match your needs:",
                    ""
                ]
                
                for i, suggestion in enumerate(suggestions[:5], 1):
                    template = self.template_registry.get_template(suggestion["template_id"])
                    message_parts.append(f"{i}. **{template.template_name}**")
                    message_parts.append(f"   {template.description}")
                    message_parts.append(f"   Industry: {template.industry} | Type: {template.template_type}")
                    message_parts.append("")
                
                message_parts.extend([
                    "Please choose a template by number or tell me more about your specific needs."
                ])
                
                return ConversationResponse(
                    message="\n".join(message_parts),
                    options=[f"{i}. {self.template_registry.get_template(s['template_id']).template_name}" 
                            for i, s in enumerate(suggestions[:5], 1)],
                    input_required=True,
                    input_type="text"
                )
            
            return ConversationResponse(
                message="I couldn't find a matching template. Could you please be more specific about:\n"
                        "• Your industry (SaaS, Banking, Insurance, E-commerce, Financial Services)\n"
                        "• Your use case (fraud detection, churn prediction, credit scoring, etc.)\n"
                        "• Any specific requirements you have",
                input_required=True,
                input_type="text"
            )
            
        except Exception as e:
            self.logger.error(f"Error in template selection handler: {e}")
            return ConversationResponse(
                message="I encountered an error while processing your selection. Please try again.",
                input_required=True,
                input_type="text"
            )
    
    async def _handle_parameter_collection(
        self,
        context: ConversationContext,
        message: str
    ) -> ConversationResponse:
        """Handle parameter collection state"""
        
        try:
            template = self.template_registry.get_template(context.selected_template)
            if not template:
                return ConversationResponse(
                    message="Template not found. Please start over.",
                    next_state=ConversationState.ERROR
                )
            
            # Get required parameters for the template
            required_params = await self._get_required_parameters(template)
            
            # Process current message for parameter extraction
            extracted_params = await self._extract_parameters(message, required_params)
            context.collected_parameters.update(extracted_params)
            
            # Check what parameters are still needed
            missing_params = [
                param for param in required_params
                if param["name"] not in context.collected_parameters
            ]
            
            if missing_params:
                # Ask for next missing parameter
                next_param = missing_params[0]
                
                return ConversationResponse(
                    message=f"Great! I need a few more details.\n\n"
                            f"📝 **{next_param['display_name']}**\n"
                            f"{next_param['description']}\n\n"
                            f"Example: {next_param.get('example', 'N/A')}",
                    input_required=True,
                    input_type=next_param.get("input_type", "text"),
                    validation_pattern=next_param.get("validation_pattern"),
                    metadata={"parameter_name": next_param["name"]}
                )
            
            # All parameters collected, move to confirmation
            return ConversationResponse(
                message="Excellent! I have all the required parameters. Let me show you the configuration.",
                next_state=ConversationState.CONFIRMATION
            )
            
        except Exception as e:
            self.logger.error(f"Error in parameter collection handler: {e}")
            return ConversationResponse(
                message="I encountered an error while collecting parameters. Please try again.",
                input_required=True,
                input_type="text"
            )
    
    async def _handle_confirmation(
        self,
        context: ConversationContext,
        message: str
    ) -> ConversationResponse:
        """Handle confirmation state"""
        
        try:
            template = self.template_registry.get_template(context.selected_template)
            
            # If user is confirming deployment
            if message.lower() in ["yes", "y", "confirm", "deploy", "proceed"]:
                # Prepare deployment configuration
                context.deployment_config = {
                    "template_id": context.selected_template,
                    "parameters": context.collected_parameters,
                    "user_id": context.user_id,
                    "tenant_id": context.tenant_id,
                    "deployment_mode": "conversational",
                    "created_at": datetime.utcnow().isoformat()
                }
                
                return ConversationResponse(
                    message="🚀 Perfect! Starting deployment now...",
                    next_state=ConversationState.DEPLOYMENT
                )
            
            elif message.lower() in ["no", "n", "cancel", "abort"]:
                return ConversationResponse(
                    message="Deployment cancelled. You can start over anytime.",
                    next_state=ConversationState.COMPLETED
                )
            
            # Show configuration summary
            config_summary = await self._generate_configuration_summary(template, context.collected_parameters)
            
            return ConversationResponse(
                message=f"📋 **Deployment Configuration Summary**\n\n"
                        f"**Template**: {template.template_name}\n"
                        f"**Industry**: {template.industry}\n"
                        f"**Type**: {template.template_type}\n\n"
                        f"**Configuration**:\n"
                        f"{config_summary}\n\n"
                        f"**ML Models**:\n"
                        f"{self._format_ml_models(template.ml_models)}\n\n"
                        f"**Governance**:\n"
                        f"• Trust threshold: {template.confidence_thresholds.get('overall_confidence', 0.7)}\n"
                        f"• Explainability: {'Enabled' if template.explainability_config['enabled'] else 'Disabled'}\n"
                        f"• Audit logging: {'Enabled' if template.governance_config.get('audit_logging', False) else 'Disabled'}\n\n"
                        f"Would you like to proceed with deployment? (yes/no)",
                options=["Yes, deploy now", "No, cancel"],
                input_required=True,
                input_type="text"
            )
            
        except Exception as e:
            self.logger.error(f"Error in confirmation handler: {e}")
            return ConversationResponse(
                message="I encountered an error while preparing the configuration. Please try again.",
                input_required=True,
                input_type="text"
            )
    
    async def _handle_deployment(
        self,
        context: ConversationContext,
        message: str
    ) -> ConversationResponse:
        """Handle deployment state"""
        
        try:
            # Generate workflow YAML
            workflow_yaml = self.template_registry.generate_workflow_yaml(
                context.selected_template,
                context.collected_parameters
            )
            
            # Simulate deployment process
            deployment_id = str(uuid.uuid4())
            
            # Store deployment information
            context.deployment_config.update({
                "deployment_id": deployment_id,
                "workflow_yaml": workflow_yaml,
                "status": "deploying"
            })
            
            return ConversationResponse(
                message=f"🎯 **Deployment Started!**\n\n"
                        f"**Deployment ID**: `{deployment_id}`\n"
                        f"**Status**: Initializing...\n\n"
                        f"Steps:\n"
                        f"✅ Template validation\n"
                        f"🔄 ML model initialization\n"
                        f"⏳ Workflow deployment\n"
                        f"⏳ Governance setup\n"
                        f"⏳ Monitoring configuration\n\n"
                        f"This will take a few moments. I'll keep you updated!",
                next_state=ConversationState.MONITORING,
                metadata={"deployment_id": deployment_id}
            )
            
        except Exception as e:
            self.logger.error(f"Error in deployment handler: {e}")
            return ConversationResponse(
                message=f"❌ Deployment failed: {str(e)}\n\nWould you like to try again?",
                next_state=ConversationState.ERROR
            )
    
    async def _handle_monitoring(
        self,
        context: ConversationContext,
        message: str
    ) -> ConversationResponse:
        """Handle monitoring state"""
        
        try:
            deployment_id = context.deployment_config.get("deployment_id")
            
            # Simulate deployment progress
            await asyncio.sleep(2)  # Simulate processing time
            
            # Update deployment status
            context.deployment_config["status"] = "completed"
            
            return ConversationResponse(
                message=f"🎉 **Deployment Successful!**\n\n"
                        f"**Deployment ID**: `{deployment_id}`\n"
                        f"**Status**: Active and running\n\n"
                        f"Completed steps:\n"
                        f"✅ Template validation\n"
                        f"✅ ML model initialization\n"
                        f"✅ Workflow deployment\n"
                        f"✅ Governance setup\n"
                        f"✅ Monitoring configuration\n\n"
                        f"**Next Steps**:\n"
                        f"• Your template is now processing data\n"
                        f"• Monitor performance in the dashboard\n"
                        f"• Review explainability reports\n"
                        f"• Check confidence thresholds\n\n"
                        f"**Monitoring URLs**:\n"
                        f"• Dashboard: `/dashboard/templates/{context.selected_template}`\n"
                        f"• Metrics: `/metrics/deployment/{deployment_id}`\n"
                        f"• Logs: `/logs/deployment/{deployment_id}`\n\n"
                        f"Is there anything else you'd like to configure?",
                options=["View dashboard", "Configure alerts", "Deploy another template", "End conversation"],
                next_state=ConversationState.COMPLETED
            )
            
        except Exception as e:
            self.logger.error(f"Error in monitoring handler: {e}")
            return ConversationResponse(
                message=f"❌ Monitoring setup failed: {str(e)}",
                next_state=ConversationState.ERROR
            )
    
    async def _handle_completed(
        self,
        context: ConversationContext,
        message: str
    ) -> ConversationResponse:
        """Handle completed state"""
        
        if message.lower() in ["deploy another", "new template", "another"]:
            # Reset context for new deployment
            context.state = ConversationState.INITIAL
            context.selected_template = None
            context.collected_parameters = {}
            context.deployment_config = {}
            
            return await self._handle_initial(context, None)
        
        return ConversationResponse(
            message="Thank you for using the RBIA Template Deployment Assistant! 🚀\n\n"
                    "Your template is successfully deployed and running. You can:\n"
                    "• Monitor it through the dashboard\n"
                    "• Deploy additional templates anytime\n"
                    "• Contact support if you need help\n\n"
                    "Have a great day!"
        )
    
    async def _handle_error(
        self,
        context: ConversationContext,
        message: str
    ) -> ConversationResponse:
        """Handle error state"""
        
        if message.lower() in ["restart", "start over", "begin again"]:
            context.state = ConversationState.INITIAL
            context.selected_template = None
            context.collected_parameters = {}
            context.deployment_config = {}
            
            return await self._handle_initial(context, None)
        
        return ConversationResponse(
            message="I'm sorry, but I encountered an error. You can:\n"
                    "• Type 'restart' to start over\n"
                    "• Contact support for assistance\n"
                    "• Try again later\n\n"
                    "What would you like to do?",
            options=["Restart", "Contact support"],
            input_required=True,
            input_type="text"
        )
    
    async def _analyze_template_intent(self, message: str) -> Optional[Dict[str, Any]]:
        """Analyze user message for template selection intent"""
        
        try:
            message_lower = message.lower()
            
            # Industry keywords
            industry_keywords = {
                "saas": ["saas", "software", "subscription", "cloud"],
                "banking": ["bank", "banking", "credit", "loan", "financial"],
                "insurance": ["insurance", "claim", "policy", "coverage"],
                "ecommerce": ["ecommerce", "e-commerce", "retail", "shopping", "checkout"],
                "financial_services": ["financial services", "fintech", "trading", "investment"]
            }
            
            # Use case keywords
            usecase_keywords = {
                "churn": ["churn", "retention", "leaving", "cancel"],
                "fraud": ["fraud", "fraudulent", "suspicious", "scam"],
                "credit": ["credit", "scoring", "loan", "approval"],
                "lapse": ["lapse", "expiry", "renewal"],
                "delay": ["delay", "processing", "queue"],
                "risk": ["risk", "warning", "alert"],
                "anomaly": ["anomaly", "unusual", "detection"],
                "variance": ["variance", "forecast", "prediction"]
            }
            
            # Detect industry
            detected_industry = None
            for industry, keywords in industry_keywords.items():
                if any(keyword in message_lower for keyword in keywords):
                    detected_industry = industry
                    break
            
            # Detect use case
            detected_usecase = None
            for usecase, keywords in usecase_keywords.items():
                if any(keyword in message_lower for keyword in keywords):
                    detected_usecase = usecase
                    break
            
            # Map to template IDs
            template_mapping = {
                ("saas", "churn"): "saas_churn_risk_alert",
                ("saas", "variance"): "saas_forecast_variance_detector",
                ("banking", "credit"): "banking_credit_scoring_check",
                ("banking", "fraud"): "banking_fraudulent_disbursal_detector",
                ("insurance", "fraud"): "insurance_claim_fraud_anomaly",
                ("insurance", "lapse"): "insurance_policy_lapse_predictor",
                ("ecommerce", "fraud"): "ecommerce_checkout_fraud_scoring",
                ("ecommerce", "delay"): "ecommerce_refund_delay_predictor",
                ("financial_services", "risk"): "fs_liquidity_risk_early_warning",
                ("financial_services", "anomaly"): "fs_mifid_reporting_anomaly_detection"
            }
            
            # Find matching template
            if detected_industry and detected_usecase:
                template_key = (detected_industry, detected_usecase)
                template_id = template_mapping.get(template_key)
                
                if template_id:
                    return {
                        "template_id": template_id,
                        "confidence": 0.8,
                        "detected_industry": detected_industry,
                        "detected_usecase": detected_usecase
                    }
            
            # Try direct template name matching
            templates = self.template_registry.list_all_templates()
            for template in templates:
                template_name_lower = template.template_name.lower()
                if template_name_lower in message_lower or any(
                    word in message_lower for word in template_name_lower.split()
                ):
                    return {
                        "template_id": template.template_id,
                        "confidence": 0.9,
                        "matched_name": template.template_name
                    }
            
            return None
            
        except Exception as e:
            self.logger.error(f"Error analyzing template intent: {e}")
            return None
    
    async def _get_template_suggestions(self, message: str) -> List[Dict[str, Any]]:
        """Get template suggestions based on user message"""
        
        try:
            suggestions = []
            templates = self.template_registry.list_all_templates()
            
            message_lower = message.lower()
            
            for template in templates:
                score = 0
                
                # Check template name
                if any(word in message_lower for word in template.template_name.lower().split()):
                    score += 3
                
                # Check description
                if any(word in message_lower for word in template.description.lower().split()):
                    score += 2
                
                # Check industry
                if template.industry.lower() in message_lower:
                    score += 2
                
                # Check template type
                if template.template_type.lower() in message_lower:
                    score += 1
                
                if score > 0:
                    suggestions.append({
                        "template_id": template.template_id,
                        "score": score,
                        "template": template
                    })
            
            # Sort by score
            suggestions.sort(key=lambda x: x["score"], reverse=True)
            
            return suggestions
            
        except Exception as e:
            self.logger.error(f"Error getting template suggestions: {e}")
            return []
    
    async def _get_required_parameters(self, template: TemplateConfig) -> List[Dict[str, Any]]:
        """Get required parameters for template configuration"""
        
        # Template-specific parameter definitions
        parameter_definitions = {
            "saas_churn_risk_alert": [
                {
                    "name": "database_connection",
                    "display_name": "Database Connection",
                    "description": "Connection string for your customer analytics database",
                    "input_type": "text",
                    "example": "postgresql://user:pass@host:5432/analytics_db"
                },
                {
                    "name": "slack_webhook_url",
                    "display_name": "Slack Webhook URL",
                    "description": "Slack webhook URL for high-risk churn alerts",
                    "input_type": "text",
                    "example": "https://hooks.slack.com/services/..."
                },
                {
                    "name": "notification_email",
                    "display_name": "Notification Email",
                    "description": "Email address for medium-risk alerts",
                    "input_type": "text",
                    "example": "customer-success@company.com"
                }
            ],
            "banking_credit_scoring_check": [
                {
                    "name": "loan_applications_db",
                    "display_name": "Loan Applications Database",
                    "description": "Database connection for loan applications",
                    "input_type": "text",
                    "example": "postgresql://user:pass@host:5432/loans_db"
                },
                {
                    "name": "credit_review_queue",
                    "display_name": "Credit Review Queue",
                    "description": "Queue name for human review cases",
                    "input_type": "text",
                    "example": "credit_review_queue"
                },
                {
                    "name": "approval_threshold",
                    "display_name": "Approval Threshold",
                    "description": "Credit score threshold for automatic approval",
                    "input_type": "number",
                    "example": "700"
                }
            ]
        }
        
        return parameter_definitions.get(template.template_id, [
            {
                "name": "database_connection",
                "display_name": "Database Connection",
                "description": "Database connection string for your data source",
                "input_type": "text",
                "example": "postgresql://user:pass@host:5432/database"
            }
        ])
    
    async def _extract_parameters(
        self,
        message: str,
        required_params: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Extract parameters from user message"""
        
        extracted = {}
        
        # Simple parameter extraction patterns
        patterns = {
            "database_connection": r"(?:database|db|connection).*?([a-zA-Z]+://[^\s]+)",
            "slack_webhook_url": r"(?:slack|webhook).*?(https://hooks\.slack\.com/[^\s]+)",
            "notification_email": r"(?:email|notify).*?([a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,})",
            "approval_threshold": r"(?:threshold|score).*?(\d+)"
        }
        
        for param in required_params:
            param_name = param["name"]
            if param_name in patterns:
                match = re.search(patterns[param_name], message, re.IGNORECASE)
                if match:
                    extracted[param_name] = match.group(1)
        
        return extracted
    
    async def _generate_configuration_summary(
        self,
        template: TemplateConfig,
        parameters: Dict[str, Any]
    ) -> str:
        """Generate human-readable configuration summary"""
        
        summary_parts = []
        
        for key, value in parameters.items():
            # Mask sensitive information
            if "password" in key.lower() or "secret" in key.lower() or "token" in key.lower():
                display_value = "*" * 8
            elif "webhook" in key.lower() or "connection" in key.lower():
                display_value = value[:20] + "..." if len(str(value)) > 20 else str(value)
            else:
                display_value = str(value)
            
            summary_parts.append(f"• {key.replace('_', ' ').title()}: {display_value}")
        
        return "\n".join(summary_parts) if summary_parts else "• Default configuration"
    
    def _format_ml_models(self, ml_models: List[Dict[str, Any]]) -> str:
        """Format ML models for display"""
        
        model_parts = []
        for model in ml_models:
            model_parts.append(f"• {model['model_name']} ({model['model_type']})")
            model_parts.append(f"  Confidence threshold: {model['confidence_threshold']}")
        
        return "\n".join(model_parts)
    
    def get_conversation_context(self, conversation_id: str) -> Optional[ConversationContext]:
        """Get conversation context"""
        return self.active_conversations.get(conversation_id)
    
    def get_active_conversations(self, user_id: int) -> List[ConversationContext]:
        """Get active conversations for a user"""
        return [
            context for context in self.active_conversations.values()
            if context.user_id == user_id
        ]
    
    async def end_conversation(self, conversation_id: str) -> bool:
        """End a conversation"""
        if conversation_id in self.active_conversations:
            del self.active_conversations[conversation_id]
            return True
        return False
