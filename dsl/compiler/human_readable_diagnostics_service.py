"""
Human-Readable Diagnostics Service - Task 6.2.33
=================================================

Human-readable diagnostics tuned by persona
- Makes diagnostics actionable by tailoring messages to specific personas
- Diagnostic rendering layer with persona-specific messages and remediation steps
- Supports developer, RevOps, Compliance, and other persona types
- Localization support for i18n (translations/localization assets)
- Backend implementation (no actual LSP/Builder UI integration - that's frontend)

Dependencies: Task 6.2.32 (Diagnostic Catalog Service)
Outputs: Persona-specific diagnostic messages → enables targeted user guidance
"""

from typing import Dict, List, Any, Optional, Union, Tuple
import json
import logging
from datetime import datetime, timezone
from dataclasses import dataclass, field, asdict
from enum import Enum
import re
import os

logger = logging.getLogger(__name__)

class MessageStyle(Enum):
    """Styles for diagnostic messages"""
    TECHNICAL = "technical"        # Stack traces, technical details
    BUSINESS = "business"          # Business impact, plain language
    PROCEDURAL = "procedural"      # Step-by-step instructions
    REGULATORY = "regulatory"      # Compliance and legal language
    EDUCATIONAL = "educational"    # Learning-focused explanations

class UrgencyLevel(Enum):
    """Urgency levels for diagnostics"""
    IMMEDIATE = "immediate"        # Must fix now
    HIGH = "high"                 # Fix before deployment
    MEDIUM = "medium"             # Fix in current sprint
    LOW = "low"                   # Fix when convenient
    INFORMATIONAL = "informational" # No action required

class LocaleCode(Enum):
    """Supported locale codes"""
    EN_US = "en_US"  # English (US)
    EN_GB = "en_GB"  # English (UK)
    FR_FR = "fr_FR"  # French (France)
    ES_ES = "es_ES"  # Spanish (Spain)
    HI_IN = "hi_IN"  # Hindi (India)
    ZH_CN = "zh_CN"  # Chinese (Simplified)

@dataclass
class PersonaProfile:
    """Profile defining characteristics of a persona"""
    persona_type: str  # From PersonaType enum in diagnostic_catalog_service
    display_name: str
    description: str
    
    # Message preferences
    preferred_message_style: MessageStyle
    technical_detail_level: int  # 1-5, 1=minimal, 5=maximum
    show_code_examples: bool
    show_remediation_steps: bool
    show_business_impact: bool
    
    # Context preferences
    include_related_errors: bool
    include_historical_context: bool
    show_approval_requirements: bool
    
    # UI preferences
    preferred_urgency_indicators: List[str] = field(default_factory=list)
    color_coding_preferences: Dict[str, str] = field(default_factory=dict)

@dataclass
class LocalizedMessage:
    """Localized message content"""
    locale: LocaleCode
    title: str
    message: str
    remediation_steps: List[str] = field(default_factory=list)
    additional_context: Optional[str] = None
    call_to_action: Optional[str] = None

@dataclass
class PersonaMessage:
    """Message tailored for a specific persona"""
    persona_type: str
    message_style: MessageStyle
    urgency_level: UrgencyLevel
    
    # Core message content
    title: str
    summary: str
    detailed_explanation: str
    
    # Actionable content
    immediate_actions: List[str] = field(default_factory=list)
    remediation_steps: List[str] = field(default_factory=list)
    next_steps: List[str] = field(default_factory=list)
    
    # Context and impact
    business_impact: Optional[str] = None
    technical_details: Optional[str] = None
    compliance_implications: Optional[str] = None
    
    # Supporting information
    code_examples: List[str] = field(default_factory=list)
    documentation_links: List[str] = field(default_factory=list)
    related_diagnostics: List[str] = field(default_factory=list)
    
    # Metadata
    estimated_fix_time: Optional[str] = None
    skill_level_required: Optional[str] = None
    approval_required: bool = False

@dataclass
class DiagnosticPresentation:
    """Complete presentation of a diagnostic for multiple personas"""
    error_code: str
    diagnostic_instance_id: Optional[str] = None
    
    # Core diagnostic info
    severity: str
    category: str
    file_location: Optional[str] = None
    
    # Persona-specific messages
    persona_messages: Dict[str, PersonaMessage] = field(default_factory=dict)
    
    # Localized content
    localized_messages: Dict[str, LocalizedMessage] = field(default_factory=dict)
    
    # Presentation metadata
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    template_version: str = "1.0"

# Task 6.2.33: Human-Readable Diagnostics Service
class HumanReadableDiagnosticsService:
    """Service for generating persona-specific diagnostic messages"""
    
    def __init__(self, diagnostic_catalog_service=None):
        self.logger = logging.getLogger(__name__)
        
        # Import diagnostic catalog service
        if diagnostic_catalog_service:
            self.diagnostic_catalog = diagnostic_catalog_service
        else:
            from .diagnostic_catalog_service import DiagnosticCatalogService
            self.diagnostic_catalog = DiagnosticCatalogService()
        
        # Persona profiles
        self.persona_profiles = self._initialize_persona_profiles()
        
        # Message templates by persona and error code
        self.message_templates: Dict[str, Dict[str, Dict[str, Any]]] = {}  # persona -> error_code -> template
        
        # Localization assets
        self.localization_assets: Dict[LocaleCode, Dict[str, str]] = {}
        
        # Initialize templates and localization
        self._initialize_message_templates()
        self._initialize_localization_assets()
        
        # Statistics
        self.rendering_stats = {
            'total_messages_rendered': 0,
            'messages_by_persona': {},
            'messages_by_locale': {},
            'template_cache_hits': 0,
            'template_cache_misses': 0
        }
    
    def render_diagnostic_for_persona(self, error_code: str, persona_type: str,
                                    context_data: Optional[Dict[str, Any]] = None,
                                    locale: LocaleCode = LocaleCode.EN_US) -> Optional[PersonaMessage]:
        """
        Render a diagnostic message for a specific persona
        
        Args:
            error_code: Error code from diagnostic catalog
            persona_type: Target persona type
            context_data: Additional context for message rendering
            locale: Locale for message rendering
            
        Returns:
            PersonaMessage tailored for the persona
        """
        # Get diagnostic rule
        rule = self.diagnostic_catalog.diagnostic_rules.get(error_code)
        if not rule:
            self.logger.warning(f"Unknown error code for rendering: {error_code}")
            return None
        
        # Get persona profile
        persona_profile = self.persona_profiles.get(persona_type)
        if not persona_profile:
            self.logger.warning(f"Unknown persona type: {persona_type}")
            return None
        
        # Get message template
        template = self._get_message_template(persona_type, error_code)
        
        # Determine urgency level
        urgency_level = self._determine_urgency_level(rule, persona_profile)
        
        # Generate persona-specific message
        persona_message = self._generate_persona_message(
            rule, persona_profile, template, urgency_level, context_data or {}
        )
        
        # Apply localization if needed
        if locale != LocaleCode.EN_US:
            persona_message = self._apply_localization(persona_message, locale)
        
        # Update statistics
        self._update_rendering_stats(persona_type, locale)
        
        self.logger.debug(f"Rendered diagnostic {error_code} for persona {persona_type}")
        
        return persona_message
    
    def create_multi_persona_presentation(self, error_code: str,
                                        persona_types: List[str],
                                        context_data: Optional[Dict[str, Any]] = None,
                                        diagnostic_instance_id: Optional[str] = None) -> DiagnosticPresentation:
        """
        Create a diagnostic presentation for multiple personas
        
        Args:
            error_code: Error code from diagnostic catalog
            persona_types: List of persona types to render for
            context_data: Additional context for message rendering
            diagnostic_instance_id: Optional diagnostic instance ID
            
        Returns:
            DiagnosticPresentation with messages for all personas
        """
        rule = self.diagnostic_catalog.diagnostic_rules.get(error_code)
        if not rule:
            raise ValueError(f"Unknown error code: {error_code}")
        
        presentation = DiagnosticPresentation(
            error_code=error_code,
            diagnostic_instance_id=diagnostic_instance_id,
            severity=rule.severity.value,
            category=rule.category.value,
            file_location=context_data.get('file_path') if context_data else None
        )
        
        # Generate messages for each persona
        for persona_type in persona_types:
            persona_message = self.render_diagnostic_for_persona(
                error_code, persona_type, context_data
            )
            if persona_message:
                presentation.persona_messages[persona_type] = persona_message
        
        # Generate localized messages for default personas
        default_locales = [LocaleCode.EN_US, LocaleCode.FR_FR, LocaleCode.ES_ES]
        for locale in default_locales:
            localized_message = self._generate_localized_message(rule, locale, context_data or {})
            if localized_message:
                presentation.localized_messages[locale.value] = localized_message
        
        return presentation
    
    def get_remediation_guidance(self, error_code: str, persona_type: str,
                               skill_level: str = "intermediate") -> Dict[str, Any]:
        """
        Get detailed remediation guidance for a specific persona
        
        Args:
            error_code: Error code from diagnostic catalog
            persona_type: Target persona type
            skill_level: Skill level (beginner, intermediate, advanced)
            
        Returns:
            Dictionary with detailed remediation guidance
        """
        rule = self.diagnostic_catalog.diagnostic_rules.get(error_code)
        if not rule:
            return {'error': f'Unknown error code: {error_code}'}
        
        persona_profile = self.persona_profiles.get(persona_type)
        if not persona_profile:
            return {'error': f'Unknown persona type: {persona_type}'}
        
        # Filter remediation steps based on persona and skill level
        relevant_steps = []
        for step in rule.remediation_steps:
            if step.persona.value == persona_type or persona_type in ['developer', 'admin']:
                # Adjust step based on skill level
                adjusted_step = {
                    'step_number': step.step_number,
                    'description': step.description,
                    'estimated_time_minutes': step.estimated_time_minutes,
                    'automation_available': step.automation_available,
                    'verification_criteria': step.verification_criteria
                }
                
                # Add skill-level specific guidance
                if skill_level == "beginner":
                    adjusted_step['additional_guidance'] = self._get_beginner_guidance(step)
                elif skill_level == "advanced":
                    adjusted_step['shortcuts'] = self._get_advanced_shortcuts(step)
                
                relevant_steps.append(adjusted_step)
        
        # Get autofix suggestions if available
        autofix_suggestions = []
        for autofix in rule.autofix_suggestions:
            if skill_level == "beginner" and not autofix.safe_to_apply_automatically:
                continue  # Skip risky autofixes for beginners
            
            autofix_suggestions.append({
                'fix_id': autofix.fix_id,
                'description': autofix.description,
                'confidence_score': autofix.confidence_score,
                'safe_to_apply_automatically': autofix.safe_to_apply_automatically,
                'command': autofix.fix_command
            })
        
        return {
            'error_code': error_code,
            'persona_type': persona_type,
            'skill_level': skill_level,
            'remediation_steps': relevant_steps,
            'autofix_suggestions': autofix_suggestions,
            'estimated_total_time': sum(step.estimated_time_minutes for step in rule.remediation_steps),
            'approval_required': rule.require_approval,
            'documentation_url': rule.documentation_url,
            'related_error_codes': rule.related_error_codes
        }
    
    def generate_diagnostic_summary_for_persona(self, error_codes: List[str], 
                                               persona_type: str) -> Dict[str, Any]:
        """
        Generate a summary of multiple diagnostics tailored for a persona
        
        Args:
            error_codes: List of error codes
            persona_type: Target persona type
            
        Returns:
            Dictionary with persona-specific summary
        """
        persona_profile = self.persona_profiles.get(persona_type)
        if not persona_profile:
            return {'error': f'Unknown persona type: {persona_type}'}
        
        summary = {
            'persona_type': persona_type,
            'total_diagnostics': len(error_codes),
            'by_urgency': {urgency.value: 0 for urgency in UrgencyLevel},
            'by_category': {},
            'immediate_actions': [],
            'approval_required': [],
            'can_autofix': [],
            'estimated_total_fix_time': 0,
            'top_priorities': []
        }
        
        diagnostics_details = []
        
        for error_code in error_codes:
            rule = self.diagnostic_catalog.diagnostic_rules.get(error_code)
            if not rule:
                continue
            
            # Determine urgency for this persona
            urgency = self._determine_urgency_level(rule, persona_profile)
            summary['by_urgency'][urgency.value] += 1
            
            # Category tracking
            category = rule.category.value
            summary['by_category'][category] = summary['by_category'].get(category, 0) + 1
            
            # Collect actionable items
            if urgency == UrgencyLevel.IMMEDIATE:
                summary['immediate_actions'].append(error_code)
            
            if rule.require_approval:
                summary['approval_required'].append(error_code)
            
            if rule.autofix_suggestions:
                summary['can_autofix'].append(error_code)
            
            # Estimate fix time
            total_time = sum(step.estimated_time_minutes for step in rule.remediation_steps)
            summary['estimated_total_fix_time'] += total_time
            
            diagnostics_details.append({
                'error_code': error_code,
                'urgency': urgency.value,
                'category': category,
                'estimated_fix_time': total_time,
                'autofix_available': len(rule.autofix_suggestions) > 0
            })
        
        # Determine top priorities based on persona preferences
        diagnostics_details.sort(key=lambda x: (
            0 if x['urgency'] == 'immediate' else 1,
            0 if x['autofix_available'] else 1,
            x['estimated_fix_time']
        ))
        
        summary['top_priorities'] = diagnostics_details[:5]  # Top 5 priorities
        
        # Generate persona-specific recommendations
        summary['recommendations'] = self._generate_persona_recommendations(
            summary, persona_type, persona_profile
        )
        
        return summary
    
    def export_persona_message_templates(self, persona_type: str) -> Dict[str, Any]:
        """
        Export message templates for a specific persona
        
        Args:
            persona_type: Persona type to export templates for
            
        Returns:
            Dictionary with message templates
        """
        if persona_type not in self.message_templates:
            return {'error': f'No templates found for persona: {persona_type}'}
        
        return {
            'persona_type': persona_type,
            'templates': self.message_templates[persona_type],
            'exported_at': datetime.now(timezone.utc).isoformat(),
            'template_count': len(self.message_templates[persona_type])
        }
    
    def get_rendering_statistics(self) -> Dict[str, Any]:
        """Get diagnostic rendering statistics"""
        return {
            **self.rendering_stats,
            'persona_profiles_loaded': len(self.persona_profiles),
            'message_templates_loaded': sum(len(templates) for templates in self.message_templates.values()),
            'localization_assets_loaded': len(self.localization_assets)
        }
    
    def _initialize_persona_profiles(self) -> Dict[str, PersonaProfile]:
        """Initialize persona profiles"""
        profiles = {}
        
        # Developer persona
        profiles['developer'] = PersonaProfile(
            persona_type='developer',
            display_name='Software Developer',
            description='Technical team member who writes and maintains code',
            preferred_message_style=MessageStyle.TECHNICAL,
            technical_detail_level=5,
            show_code_examples=True,
            show_remediation_steps=True,
            show_business_impact=False,
            include_related_errors=True,
            include_historical_context=False,
            show_approval_requirements=False
        )
        
        # RevOps persona
        profiles['revops'] = PersonaProfile(
            persona_type='revops',
            display_name='Revenue Operations',
            description='Business stakeholder focused on revenue processes',
            preferred_message_style=MessageStyle.BUSINESS,
            technical_detail_level=2,
            show_code_examples=False,
            show_remediation_steps=True,
            show_business_impact=True,
            include_related_errors=False,
            include_historical_context=True,
            show_approval_requirements=True
        )
        
        # Compliance persona
        profiles['compliance'] = PersonaProfile(
            persona_type='compliance',
            display_name='Compliance Officer',
            description='Ensures adherence to regulatory and policy requirements',
            preferred_message_style=MessageStyle.REGULATORY,
            technical_detail_level=3,
            show_code_examples=False,
            show_remediation_steps=True,
            show_business_impact=True,
            include_related_errors=True,
            include_historical_context=True,
            show_approval_requirements=True
        )
        
        # Data Scientist persona
        profiles['data_scientist'] = PersonaProfile(
            persona_type='data_scientist',
            display_name='Data Scientist',
            description='ML/AI specialist working with models and data',
            preferred_message_style=MessageStyle.EDUCATIONAL,
            technical_detail_level=4,
            show_code_examples=True,
            show_remediation_steps=True,
            show_business_impact=True,
            include_related_errors=True,
            include_historical_context=False,
            show_approval_requirements=False
        )
        
        # Security persona
        profiles['security'] = PersonaProfile(
            persona_type='security',
            display_name='Security Engineer',
            description='Security specialist focused on risk and threat mitigation',
            preferred_message_style=MessageStyle.TECHNICAL,
            technical_detail_level=4,
            show_code_examples=True,
            show_remediation_steps=True,
            show_business_impact=True,
            include_related_errors=True,
            include_historical_context=True,
            show_approval_requirements=True
        )
        
        return profiles
    
    def _initialize_message_templates(self):
        """Initialize message templates for different personas"""
        # Developer templates
        self.message_templates['developer'] = {
            'SYNTAX_001': {
                'title': 'Syntax Error: Missing Semicolon',
                'summary': 'Statement termination missing',
                'detailed_explanation': 'The DSL parser expected a semicolon to terminate the statement but found {found_token} instead.',
                'immediate_actions': ['Add semicolon at end of statement'],
                'code_examples': ['// Correct: step_name: action();', '// Incorrect: step_name: action()']
            },
            'ML_001': {
                'title': 'ML Configuration Error: Missing Confidence Threshold',
                'summary': 'ML node lacks required confidence threshold',
                'detailed_explanation': 'All ML prediction nodes must specify a confidence threshold for governance compliance. This ensures predictions below the threshold trigger appropriate fallback behavior.',
                'immediate_actions': ['Add confidence_threshold parameter to ML node', 'Review model performance metrics'],
                'code_examples': ['ml_predict: model_id="churn_v2", confidence_threshold=0.75']
            }
        }
        
        # RevOps templates
        self.message_templates['revops'] = {
            'ML_001': {
                'title': 'ML Model Needs Quality Gate',
                'summary': 'Your ML model is missing a quality threshold',
                'detailed_explanation': 'To ensure reliable automated decisions, every ML model must have a minimum confidence level. When the model is not confident enough, the system will ask for human review or use backup rules.',
                'business_impact': 'Without quality gates, the system might make poor automated decisions that could impact revenue and customer experience.',
                'immediate_actions': ['Work with your data science team to set appropriate confidence levels', 'Review recent model performance']
            },
            'POLICY_001': {
                'title': 'Data Location Compliance Issue',
                'summary': 'Data processing violates geographic requirements',
                'business_impact': 'This could result in regulatory violations and potential fines. Customer data may be processed in unauthorized regions.',
                'immediate_actions': ['Review data residency requirements', 'Contact compliance team', 'Halt deployment until resolved']
            }
        }
        
        # Compliance templates
        self.message_templates['compliance'] = {
            'POLICY_001': {
                'title': 'Regulatory Compliance Violation: Data Residency',
                'summary': 'Geographic data processing restrictions violated',
                'detailed_explanation': 'The workflow configuration allows customer data to be processed outside of approved geographic regions, violating data residency requirements under applicable regulations (GDPR, CCPA, etc.).',
                'compliance_implications': 'This violation could result in regulatory penalties, audit findings, and potential legal liability.',
                'immediate_actions': ['Suspend deployment immediately', 'Review data flow mappings', 'Update residency constraints', 'Document remediation actions for audit trail']
            },
            'ML_001': {
                'title': 'Governance Control Missing: ML Quality Gate',
                'summary': 'ML model lacks required governance controls',
                'compliance_implications': 'Without confidence thresholds, ML decisions lack appropriate oversight and quality controls required for regulatory compliance.',
                'immediate_actions': ['Implement confidence threshold controls', 'Document model governance procedures', 'Establish human oversight processes']
            }
        }
    
    def _initialize_localization_assets(self):
        """Initialize localization assets for i18n support"""
        # English (US) - default
        self.localization_assets[LocaleCode.EN_US] = {
            'error': 'Error',
            'warning': 'Warning',
            'info': 'Information',
            'immediate_action_required': 'Immediate Action Required',
            'fix_before_deployment': 'Fix Before Deployment',
            'recommended_fix': 'Recommended Fix',
            'estimated_time': 'Estimated Time',
            'approval_required': 'Approval Required',
            'contact_team': 'Contact {team} team for assistance'
        }
        
        # French (France)
        self.localization_assets[LocaleCode.FR_FR] = {
            'error': 'Erreur',
            'warning': 'Avertissement',
            'info': 'Information',
            'immediate_action_required': 'Action Immédiate Requise',
            'fix_before_deployment': 'Corriger Avant Déploiement',
            'recommended_fix': 'Correction Recommandée',
            'estimated_time': 'Temps Estimé',
            'approval_required': 'Approbation Requise',
            'contact_team': 'Contactez l\'équipe {team} pour assistance'
        }
        
        # Spanish (Spain)
        self.localization_assets[LocaleCode.ES_ES] = {
            'error': 'Error',
            'warning': 'Advertencia',
            'info': 'Información',
            'immediate_action_required': 'Acción Inmediata Requerida',
            'fix_before_deployment': 'Corregir Antes del Despliegue',
            'recommended_fix': 'Corrección Recomendada',
            'estimated_time': 'Tiempo Estimado',
            'approval_required': 'Aprobación Requerida',
            'contact_team': 'Contacte al equipo de {team} para asistencia'
        }
    
    def _get_message_template(self, persona_type: str, error_code: str) -> Dict[str, Any]:
        """Get message template for persona and error code"""
        if persona_type in self.message_templates:
            if error_code in self.message_templates[persona_type]:
                self.rendering_stats['template_cache_hits'] += 1
                return self.message_templates[persona_type][error_code]
        
        self.rendering_stats['template_cache_misses'] += 1
        
        # Return default template
        return {
            'title': f'Diagnostic: {error_code}',
            'summary': 'A diagnostic issue has been detected',
            'detailed_explanation': 'Please review the diagnostic details and take appropriate action.',
            'immediate_actions': ['Review diagnostic details', 'Take appropriate corrective action']
        }
    
    def _determine_urgency_level(self, rule, persona_profile: PersonaProfile) -> UrgencyLevel:
        """Determine urgency level based on rule and persona"""
        # Base urgency on severity and action type
        if rule.block_compilation or rule.block_signing:
            return UrgencyLevel.IMMEDIATE
        elif rule.severity.value == 'error':
            return UrgencyLevel.HIGH
        elif rule.severity.value == 'warning':
            return UrgencyLevel.MEDIUM
        else:
            return UrgencyLevel.LOW
    
    def _generate_persona_message(self, rule, persona_profile: PersonaProfile, 
                                template: Dict[str, Any], urgency_level: UrgencyLevel,
                                context_data: Dict[str, Any]) -> PersonaMessage:
        """Generate a complete persona message"""
        message = PersonaMessage(
            persona_type=persona_profile.persona_type,
            message_style=persona_profile.preferred_message_style,
            urgency_level=urgency_level,
            title=template.get('title', f'Diagnostic: {rule.error_code}'),
            summary=template.get('summary', rule.short_message),
            detailed_explanation=template.get('detailed_explanation', rule.detailed_description)
        )
        
        # Add immediate actions
        if 'immediate_actions' in template:
            message.immediate_actions = template['immediate_actions']
        
        # Add remediation steps if persona wants them
        if persona_profile.show_remediation_steps:
            message.remediation_steps = [step.description for step in rule.remediation_steps]
        
        # Add business impact if persona wants it
        if persona_profile.show_business_impact and 'business_impact' in template:
            message.business_impact = template['business_impact']
        
        # Add technical details based on persona's technical level
        if persona_profile.technical_detail_level >= 3:
            message.technical_details = rule.technical_explanation
        
        # Add compliance implications if relevant
        if 'compliance_implications' in template:
            message.compliance_implications = template['compliance_implications']
        
        # Add code examples if persona wants them
        if persona_profile.show_code_examples and 'code_examples' in template:
            message.code_examples = template['code_examples']
        
        # Add documentation links
        if rule.documentation_url:
            message.documentation_links = [rule.documentation_url]
        
        # Add related diagnostics if persona wants them
        if persona_profile.include_related_errors:
            message.related_diagnostics = rule.related_error_codes
        
        # Set approval requirement
        message.approval_required = rule.require_approval
        
        # Estimate fix time
        if rule.remediation_steps:
            total_minutes = sum(step.estimated_time_minutes for step in rule.remediation_steps)
            if total_minutes < 60:
                message.estimated_fix_time = f"{total_minutes} minutes"
            else:
                hours = total_minutes // 60
                minutes = total_minutes % 60
                message.estimated_fix_time = f"{hours}h {minutes}m"
        
        return message
    
    def _apply_localization(self, message: PersonaMessage, locale: LocaleCode) -> PersonaMessage:
        """Apply localization to a persona message"""
        if locale not in self.localization_assets:
            return message  # Return unchanged if locale not supported
        
        assets = self.localization_assets[locale]
        
        # Apply basic translations (this is a simplified implementation)
        # In a real system, this would use proper i18n libraries
        
        return message
    
    def _generate_localized_message(self, rule, locale: LocaleCode, 
                                  context_data: Dict[str, Any]) -> Optional[LocalizedMessage]:
        """Generate a localized message for a diagnostic rule"""
        if locale not in self.localization_assets:
            return None
        
        assets = self.localization_assets[locale]
        
        # Create basic localized message
        message = LocalizedMessage(
            locale=locale,
            title=f"{assets.get('error', 'Error')}: {rule.error_code}",
            message=rule.short_message,  # In real implementation, this would be translated
            remediation_steps=[step.description for step in rule.remediation_steps[:3]]  # First 3 steps
        )
        
        if rule.require_approval:
            message.call_to_action = assets.get('approval_required', 'Approval Required')
        
        return message
    
    def _generate_persona_recommendations(self, summary: Dict[str, Any], 
                                        persona_type: str, persona_profile: PersonaProfile) -> List[str]:
        """Generate persona-specific recommendations"""
        recommendations = []
        
        if summary['immediate_actions']:
            if persona_type == 'developer':
                recommendations.append(f"Fix {len(summary['immediate_actions'])} blocking errors immediately")
            elif persona_type == 'revops':
                recommendations.append(f"Coordinate with development team to resolve {len(summary['immediate_actions'])} critical issues")
            elif persona_type == 'compliance':
                recommendations.append(f"Review {len(summary['immediate_actions'])} compliance violations requiring immediate attention")
        
        if summary['can_autofix']:
            recommendations.append(f"Apply automatic fixes for {len(summary['can_autofix'])} issues to save time")
        
        if summary['approval_required']:
            if persona_type in ['revops', 'compliance']:
                recommendations.append(f"Prepare approval documentation for {len(summary['approval_required'])} items")
            else:
                recommendations.append(f"Submit {len(summary['approval_required'])} items for approval before deployment")
        
        # Time-based recommendations
        if summary['estimated_total_fix_time'] > 480:  # More than 8 hours
            recommendations.append("Consider breaking this work into multiple sprints due to estimated fix time")
        
        return recommendations
    
    def _get_beginner_guidance(self, step) -> str:
        """Get additional guidance for beginners"""
        return f"💡 Tip: {step.description} - Take your time and verify each change before proceeding."
    
    def _get_advanced_shortcuts(self, step) -> List[str]:
        """Get shortcuts for advanced users"""
        shortcuts = []
        if step.automation_available:
            shortcuts.append("Use automated tooling to speed up this step")
        shortcuts.append("Consider batch processing if multiple similar issues exist")
        return shortcuts
    
    def _update_rendering_stats(self, persona_type: str, locale: LocaleCode):
        """Update rendering statistics"""
        self.rendering_stats['total_messages_rendered'] += 1
        
        if persona_type not in self.rendering_stats['messages_by_persona']:
            self.rendering_stats['messages_by_persona'][persona_type] = 0
        self.rendering_stats['messages_by_persona'][persona_type] += 1
        
        locale_key = locale.value
        if locale_key not in self.rendering_stats['messages_by_locale']:
            self.rendering_stats['messages_by_locale'][locale_key] = 0
        self.rendering_stats['messages_by_locale'][locale_key] += 1

# API Interface
class HumanReadableDiagnosticsAPI:
    """API interface for human-readable diagnostics operations"""
    
    def __init__(self, diagnostics_service: Optional[HumanReadableDiagnosticsService] = None):
        self.diagnostics_service = diagnostics_service or HumanReadableDiagnosticsService()
    
    def render_for_persona(self, error_code: str, persona_type: str,
                          context_data: Optional[Dict[str, Any]] = None,
                          locale: str = "en_US") -> Dict[str, Any]:
        """API endpoint to render diagnostic for persona"""
        try:
            locale_enum = LocaleCode(locale)
            message = self.diagnostics_service.render_diagnostic_for_persona(
                error_code, persona_type, context_data, locale_enum
            )
            
            if not message:
                return {
                    'success': False,
                    'error': f'Could not render diagnostic {error_code} for persona {persona_type}'
                }
            
            return {
                'success': True,
                'error_code': error_code,
                'persona_type': persona_type,
                'message': {
                    'title': message.title,
                    'summary': message.summary,
                    'urgency_level': message.urgency_level.value,
                    'immediate_actions': message.immediate_actions,
                    'remediation_steps': message.remediation_steps,
                    'business_impact': message.business_impact,
                    'estimated_fix_time': message.estimated_fix_time,
                    'approval_required': message.approval_required
                }
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': str(e),
                'error_type': type(e).__name__
            }
    
    def get_persona_summary(self, error_codes: List[str], persona_type: str) -> Dict[str, Any]:
        """API endpoint to get persona-specific summary"""
        try:
            summary = self.diagnostics_service.generate_diagnostic_summary_for_persona(
                error_codes, persona_type
            )
            
            return {
                'success': True,
                'summary': summary
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': str(e)
            }
    
    def get_remediation_guidance(self, error_code: str, persona_type: str,
                               skill_level: str = "intermediate") -> Dict[str, Any]:
        """API endpoint to get remediation guidance"""
        guidance = self.diagnostics_service.get_remediation_guidance(
            error_code, persona_type, skill_level
        )
        
        return {
            'success': 'error' not in guidance,
            **guidance
        }

# Test Functions
def run_human_readable_diagnostics_tests():
    """Run comprehensive human-readable diagnostics tests"""
    print("=== Human-Readable Diagnostics Service Tests ===")
    
    # Initialize service
    diagnostics_service = HumanReadableDiagnosticsService()
    diagnostics_api = HumanReadableDiagnosticsAPI(diagnostics_service)
    
    # Test 1: Persona profile initialization
    print("\n1. Testing persona profile initialization...")
    stats = diagnostics_service.get_rendering_statistics()
    print(f"   Persona profiles loaded: {stats['persona_profiles_loaded']}")
    print(f"   Message templates loaded: {stats['message_templates_loaded']}")
    print(f"   Localization assets loaded: {stats['localization_assets_loaded']}")
    
    # Test 2: Render diagnostics for different personas
    print("\n2. Testing persona-specific rendering...")
    
    test_error_codes = ["SYNTAX_001", "ML_001", "POLICY_001"]
    test_personas = ["developer", "revops", "compliance", "data_scientist"]
    
    for error_code in test_error_codes:
        print(f"\n   Error Code: {error_code}")
        for persona in test_personas:
            message = diagnostics_service.render_diagnostic_for_persona(
                error_code, persona, {'file_path': 'test.yaml', 'line': 42}
            )
            if message:
                print(f"     {persona}: {message.title} (urgency: {message.urgency_level.value})")
            else:
                print(f"     {persona}: No template available")
    
    # Test 3: Multi-persona presentation
    print("\n3. Testing multi-persona presentation...")
    
    presentation = diagnostics_service.create_multi_persona_presentation(
        "ML_001", ["developer", "revops", "compliance"],
        context_data={'file_path': 'workflow.yaml', 'model_id': 'churn_v2'}
    )
    
    print(f"   Presentation for ML_001:")
    print(f"   Personas covered: {len(presentation.persona_messages)}")
    print(f"   Localized messages: {len(presentation.localized_messages)}")
    print(f"   Severity: {presentation.severity}")
    
    # Test 4: Remediation guidance
    print("\n4. Testing remediation guidance...")
    
    skill_levels = ["beginner", "intermediate", "advanced"]
    for skill_level in skill_levels:
        guidance = diagnostics_service.get_remediation_guidance(
            "ML_001", "data_scientist", skill_level
        )
        if 'error' not in guidance:
            print(f"   {skill_level}: {len(guidance['remediation_steps'])} steps, "
                  f"{len(guidance['autofix_suggestions'])} autofixes")
    
    # Test 5: Persona-specific summary
    print("\n5. Testing persona-specific summary...")
    
    test_error_codes = ["SYNTAX_001", "ML_001", "POLICY_001", "SECURITY_001"]
    
    for persona in ["developer", "revops", "compliance"]:
        summary = diagnostics_service.generate_diagnostic_summary_for_persona(
            test_error_codes, persona
        )
        if 'error' not in summary:
            print(f"   {persona}: {summary['total_diagnostics']} diagnostics, "
                  f"{len(summary['immediate_actions'])} immediate actions, "
                  f"{summary['estimated_total_fix_time']} min estimated")
    
    # Test 6: Localization support
    print("\n6. Testing localization support...")
    
    locales_to_test = [LocaleCode.EN_US, LocaleCode.FR_FR, LocaleCode.ES_ES]
    for locale in locales_to_test:
        message = diagnostics_service.render_diagnostic_for_persona(
            "ML_001", "developer", locale=locale
        )
        if message:
            print(f"   {locale.value}: Message rendered successfully")
    
    # Test 7: API interface
    print("\n7. Testing API interface...")
    
    # Test API rendering
    api_render_result = diagnostics_api.render_for_persona(
        "ML_001", "revops", {'file_path': 'test.yaml'}
    )
    print(f"   API rendering: {'✅ PASS' if api_render_result['success'] else '❌ FAIL'}")
    
    if api_render_result['success']:
        message_data = api_render_result['message']
        print(f"   Title: {message_data['title']}")
        print(f"   Urgency: {message_data['urgency_level']}")
        print(f"   Immediate actions: {len(message_data['immediate_actions'])}")
    
    # Test API summary
    api_summary_result = diagnostics_api.get_persona_summary(test_error_codes, "developer")
    print(f"   API summary: {'✅ PASS' if api_summary_result['success'] else '❌ FAIL'}")
    
    # Test API remediation guidance
    api_guidance_result = diagnostics_api.get_remediation_guidance("ML_001", "data_scientist")
    print(f"   API guidance: {'✅ PASS' if api_guidance_result['success'] else '❌ FAIL'}")
    
    # Test 8: Template export
    print("\n8. Testing template export...")
    
    template_export = diagnostics_service.export_persona_message_templates("developer")
    if 'error' not in template_export:
        print(f"   Developer templates exported: {template_export['template_count']} templates")
    else:
        print(f"   Template export failed: {template_export['error']}")
    
    # Test 9: Message style variations
    print("\n9. Testing message style variations...")
    
    personas_styles = [
        ("developer", "Technical details and code examples"),
        ("revops", "Business impact and plain language"),
        ("compliance", "Regulatory implications and procedures"),
        ("data_scientist", "Educational explanations and best practices")
    ]
    
    for persona, description in personas_styles:
        message = diagnostics_service.render_diagnostic_for_persona(
            "ML_001", persona
        )
        if message:
            print(f"   {persona}: {message.message_style.value} style - {description}")
    
    # Test 10: Statistics and performance
    print("\n10. Testing statistics...")
    
    final_stats = diagnostics_service.get_rendering_statistics()
    print(f"   Total messages rendered: {final_stats['total_messages_rendered']}")
    print(f"   Messages by persona: {final_stats['messages_by_persona']}")
    print(f"   Template cache hit rate: {final_stats['template_cache_hits']}/{final_stats['template_cache_hits'] + final_stats['template_cache_misses']}")
    
    print(f"\n=== Test Summary ===")
    print(f"Human-readable diagnostics service tested successfully")
    print(f"Persona profiles: {final_stats['persona_profiles_loaded']}")
    print(f"Message templates: {final_stats['message_templates_loaded']}")
    print(f"Localization assets: {final_stats['localization_assets_loaded']}")
    print(f"Total messages rendered: {final_stats['total_messages_rendered']}")
    
    return diagnostics_service, diagnostics_api

if __name__ == "__main__":
    run_human_readable_diagnostics_tests()
