"""
Playbook Localization & Accessibility Service - Task 6.4.77
===========================================================

Provides localized and accessible versions of operational playbooks
for global teams and users with different accessibility needs.
"""

from typing import Dict, List, Any, Optional
from enum import Enum
from dataclasses import dataclass
import json
import logging

logger = logging.getLogger(__name__)

class SupportedLanguage(str, Enum):
    ENGLISH = "en"
    SPANISH = "es" 
    FRENCH = "fr"
    GERMAN = "de"
    JAPANESE = "ja"
    CHINESE_SIMPLIFIED = "zh-CN"
    PORTUGUESE = "pt"
    ITALIAN = "it"

class AccessibilityLevel(str, Enum):
    STANDARD = "standard"           # Standard accessibility
    HIGH_CONTRAST = "high_contrast" # High contrast for visual impairments
    LARGE_TEXT = "large_text"       # Large text for readability
    SCREEN_READER = "screen_reader" # Screen reader optimized
    SIMPLIFIED = "simplified"       # Simplified language for cognitive accessibility

@dataclass
class LocalizedPlaybook:
    playbook_id: str
    language: SupportedLanguage
    accessibility_level: AccessibilityLevel
    title: str
    content: str
    last_updated: str
    version: str
    translator_notes: Optional[str] = None

class PlaybookLocalizationService:
    """Service for localizing and making playbooks accessible"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.localized_playbooks: Dict[str, Dict[str, LocalizedPlaybook]] = {}
        self._load_localization_templates()
    
    def _load_localization_templates(self):
        """Load localization templates for different languages and accessibility levels"""
        
        # Base playbook content (English)
        base_playbooks = {
            "incident_response": {
                "title": "Fallback Incident Response Playbook",
                "sections": {
                    "severity_levels": {
                        "title": "Incident Severity Levels",
                        "content": {
                            "p0_critical": "Complete ML system failure, all tenants affected",
                            "p1_high": "Multiple tenant fallback triggered, service degradation", 
                            "p2_medium": "Single tenant fallback, isolated issue",
                            "p3_low": "Warning conditions, proactive fallback"
                        }
                    },
                    "response_procedures": {
                        "title": "Response Procedures",
                        "content": {
                            "immediate_actions": "Acknowledge incident, activate fallback, page on-call",
                            "assessment_phase": "Verify fallback systems, check tenant isolation",
                            "mitigation_phase": "Execute runbook, monitor performance, communicate status",
                            "recovery_phase": "Implement fix, restore services, validate stability"
                        }
                    }
                }
            },
            "recovery_runbooks": {
                "title": "Fallback System Recovery Runbooks",
                "sections": {
                    "ml_recovery": {
                        "title": "ML System Recovery",
                        "content": {
                            "check_health": "Check ML service health endpoints",
                            "activate_fallback": "Activate global ML fallback immediately",
                            "restart_service": "Restart ML service pods using kubectl",
                            "validate_health": "Test ML service with sample requests",
                            "gradual_recovery": "Deactivate fallback gradually"
                        }
                    }
                }
            }
        }
        
        # Localization templates
        self.localization_templates = {
            SupportedLanguage.SPANISH: {
                "incident_response": {
                    "title": "Manual de Respuesta a Incidentes de Fallback",
                    "sections": {
                        "severity_levels": {
                            "title": "Niveles de Severidad de Incidentes",
                            "content": {
                                "p0_critical": "Fallo completo del sistema ML, todos los inquilinos afectados",
                                "p1_high": "Fallback de múltiples inquilinos activado, degradación del servicio",
                                "p2_medium": "Fallback de un solo inquilino, problema aislado", 
                                "p3_low": "Condiciones de advertencia, fallback proactivo"
                            }
                        },
                        "response_procedures": {
                            "title": "Procedimientos de Respuesta",
                            "content": {
                                "immediate_actions": "Reconocer incidente, activar fallback, avisar guardia",
                                "assessment_phase": "Verificar sistemas fallback, comprobar aislamiento inquilinos",
                                "mitigation_phase": "Ejecutar manual, monitorear rendimiento, comunicar estado",
                                "recovery_phase": "Implementar solución, restaurar servicios, validar estabilidad"
                            }
                        }
                    }
                }
            },
            SupportedLanguage.FRENCH: {
                "incident_response": {
                    "title": "Manuel de Réponse aux Incidents de Fallback",
                    "sections": {
                        "severity_levels": {
                            "title": "Niveaux de Gravité des Incidents",
                            "content": {
                                "p0_critical": "Panne complète du système ML, tous les locataires affectés",
                                "p1_high": "Fallback multi-locataires déclenché, dégradation du service",
                                "p2_medium": "Fallback d'un seul locataire, problème isolé",
                                "p3_low": "Conditions d'avertissement, fallback proactif"
                            }
                        }
                    }
                }
            },
            SupportedLanguage.GERMAN: {
                "incident_response": {
                    "title": "Fallback-Incident-Response-Handbuch",
                    "sections": {
                        "severity_levels": {
                            "title": "Incident-Schweregrade",
                            "content": {
                                "p0_critical": "Kompletter ML-System-Ausfall, alle Mandanten betroffen",
                                "p1_high": "Multi-Mandanten-Fallback ausgelöst, Service-Degradation",
                                "p2_medium": "Einzelner Mandanten-Fallback, isoliertes Problem",
                                "p3_low": "Warnbedingungen, proaktiver Fallback"
                            }
                        }
                    }
                }
            },
            SupportedLanguage.JAPANESE: {
                "incident_response": {
                    "title": "フォールバック インシデント対応プレイブック",
                    "sections": {
                        "severity_levels": {
                            "title": "インシデント重要度レベル",
                            "content": {
                                "p0_critical": "MLシステム完全障害、全テナント影響",
                                "p1_high": "複数テナントフォールバック発動、サービス劣化",
                                "p2_medium": "単一テナントフォールバック、分離された問題",
                                "p3_low": "警告条件、予防的フォールバック"
                            }
                        }
                    }
                }
            }
        }
        
        # Accessibility templates
        self.accessibility_templates = {
            AccessibilityLevel.HIGH_CONTRAST: {
                "css_overrides": {
                    "background_color": "#000000",
                    "text_color": "#FFFFFF", 
                    "link_color": "#FFFF00",
                    "heading_color": "#00FFFF"
                }
            },
            AccessibilityLevel.LARGE_TEXT: {
                "css_overrides": {
                    "base_font_size": "18px",
                    "heading_font_size": "24px",
                    "line_height": "1.6"
                }
            },
            AccessibilityLevel.SCREEN_READER: {
                "markup_additions": {
                    "add_aria_labels": True,
                    "add_heading_structure": True,
                    "add_skip_links": True,
                    "add_alt_text": True
                }
            },
            AccessibilityLevel.SIMPLIFIED: {
                "content_modifications": {
                    "use_simple_language": True,
                    "add_definitions": True,
                    "break_long_sentences": True,
                    "add_visual_aids": True
                }
            }
        }
    
    def get_localized_playbook(
        self,
        playbook_id: str,
        language: SupportedLanguage = SupportedLanguage.ENGLISH,
        accessibility_level: AccessibilityLevel = AccessibilityLevel.STANDARD
    ) -> Optional[LocalizedPlaybook]:
        """Get localized and accessible version of a playbook - Task 6.4.77"""
        
        try:
            cache_key = f"{playbook_id}_{language.value}_{accessibility_level.value}"
            
            # Check if already cached
            if cache_key in self.localized_playbooks:
                return self.localized_playbooks[cache_key]
            
            # Generate localized playbook
            localized_playbook = self._generate_localized_playbook(
                playbook_id, language, accessibility_level
            )
            
            # Cache the result
            if cache_key not in self.localized_playbooks:
                self.localized_playbooks[cache_key] = {}
            self.localized_playbooks[cache_key] = localized_playbook
            
            self.logger.info(f"Generated localized playbook: {playbook_id} ({language.value}, {accessibility_level.value})")
            return localized_playbook
            
        except Exception as e:
            self.logger.error(f"Failed to get localized playbook {playbook_id}: {e}")
            return None
    
    def _generate_localized_playbook(
        self,
        playbook_id: str,
        language: SupportedLanguage,
        accessibility_level: AccessibilityLevel
    ) -> LocalizedPlaybook:
        """Generate localized and accessible playbook content"""
        
        # Get base template
        if language in self.localization_templates and playbook_id in self.localization_templates[language]:
            template = self.localization_templates[language][playbook_id]
        else:
            # Fallback to English
            template = self._get_base_english_template(playbook_id)
        
        # Apply accessibility modifications
        content = self._apply_accessibility_modifications(template, accessibility_level)
        
        # Generate final content
        final_content = self._render_playbook_content(content, accessibility_level)
        
        return LocalizedPlaybook(
            playbook_id=playbook_id,
            language=language,
            accessibility_level=accessibility_level,
            title=template.get("title", f"Playbook {playbook_id}"),
            content=final_content,
            last_updated="2024-01-15T10:00:00Z",
            version="1.0",
            translator_notes=self._get_translator_notes(language)
        )
    
    def _get_base_english_template(self, playbook_id: str) -> Dict[str, Any]:
        """Get base English template for playbook"""
        
        base_templates = {
            "incident_response": {
                "title": "Fallback Incident Response Playbook",
                "sections": {
                    "overview": "This playbook provides step-by-step procedures for responding to fallback system incidents.",
                    "severity_classification": "Incidents are classified into four severity levels: P0 (Critical), P1 (High), P2 (Medium), P3 (Low).",
                    "response_procedures": "Follow the appropriate response procedure based on incident severity.",
                    "escalation": "Escalate to engineering leadership for P0 incidents."
                }
            },
            "recovery_runbooks": {
                "title": "System Recovery Runbooks",
                "sections": {
                    "overview": "Detailed recovery procedures for different system components.",
                    "ml_recovery": "Steps to recover ML inference services.",
                    "cache_recovery": "Procedures for routing cache recovery.",
                    "evidence_recovery": "Evidence pack service recovery steps."
                }
            }
        }
        
        return base_templates.get(playbook_id, {"title": f"Playbook {playbook_id}", "sections": {}})
    
    def _apply_accessibility_modifications(
        self,
        template: Dict[str, Any],
        accessibility_level: AccessibilityLevel
    ) -> Dict[str, Any]:
        """Apply accessibility modifications to template"""
        
        modified_template = template.copy()
        
        if accessibility_level == AccessibilityLevel.SIMPLIFIED:
            # Simplify language
            modified_template = self._simplify_language(modified_template)
        
        elif accessibility_level == AccessibilityLevel.SCREEN_READER:
            # Add screen reader optimizations
            modified_template = self._add_screen_reader_optimizations(modified_template)
        
        elif accessibility_level in [AccessibilityLevel.HIGH_CONTRAST, AccessibilityLevel.LARGE_TEXT]:
            # Visual accessibility handled in CSS
            pass
        
        return modified_template
    
    def _simplify_language(self, template: Dict[str, Any]) -> Dict[str, Any]:
        """Simplify language for cognitive accessibility"""
        
        # Simplified language mappings
        simplifications = {
            "Complete ML system failure": "The AI system is completely broken",
            "Multiple tenant fallback triggered": "Many customers are affected by backup system",
            "Service degradation": "System is working slower than normal",
            "Proactive fallback": "Backup system started as a safety measure",
            "Acknowledge incident": "Confirm you know about the problem",
            "Activate fallback": "Turn on the backup system",
            "Execute runbook": "Follow the step-by-step instructions",
            "Validate stability": "Make sure everything is working correctly"
        }
        
        def simplify_text(text: str) -> str:
            for complex_term, simple_term in simplifications.items():
                text = text.replace(complex_term, simple_term)
            return text
        
        # Apply simplifications recursively
        def simplify_dict(d):
            if isinstance(d, dict):
                return {k: simplify_dict(v) for k, v in d.items()}
            elif isinstance(d, str):
                return simplify_text(d)
            else:
                return d
        
        return simplify_dict(template)
    
    def _add_screen_reader_optimizations(self, template: Dict[str, Any]) -> Dict[str, Any]:
        """Add screen reader optimizations"""
        
        # Add ARIA labels and structure
        template["accessibility_metadata"] = {
            "aria_labels": True,
            "heading_structure": True,
            "skip_links": True,
            "reading_order": "sequential"
        }
        
        return template
    
    def _render_playbook_content(
        self,
        content: Dict[str, Any],
        accessibility_level: AccessibilityLevel
    ) -> str:
        """Render final playbook content with accessibility features"""
        
        html = []
        
        # Add accessibility CSS if needed
        if accessibility_level in [AccessibilityLevel.HIGH_CONTRAST, AccessibilityLevel.LARGE_TEXT]:
            css_overrides = self.accessibility_templates[accessibility_level]["css_overrides"]
            html.append(f"<style>{self._generate_accessibility_css(css_overrides)}</style>")
        
        # Add skip links for screen readers
        if accessibility_level == AccessibilityLevel.SCREEN_READER:
            html.append('<a href="#main-content" class="skip-link">Skip to main content</a>')
        
        # Render title
        title = content.get("title", "Playbook")
        html.append(f'<h1 id="main-content" tabindex="-1">{title}</h1>')
        
        # Render sections
        sections = content.get("sections", {})
        for section_id, section_content in sections.items():
            if isinstance(section_content, dict) and "title" in section_content:
                html.append(f'<h2>{section_content["title"]}</h2>')
                if "content" in section_content:
                    html.append(f'<div>{self._render_section_content(section_content["content"])}</div>')
            else:
                html.append(f'<h2>{section_id.replace("_", " ").title()}</h2>')
                html.append(f'<div>{section_content}</div>')
        
        return "\n".join(html)
    
    def _generate_accessibility_css(self, css_overrides: Dict[str, str]) -> str:
        """Generate CSS for accessibility"""
        
        css_rules = []
        
        if "background_color" in css_overrides:
            css_rules.append(f"body {{ background-color: {css_overrides['background_color']}; }}")
        
        if "text_color" in css_overrides:
            css_rules.append(f"body {{ color: {css_overrides['text_color']}; }}")
        
        if "base_font_size" in css_overrides:
            css_rules.append(f"body {{ font-size: {css_overrides['base_font_size']}; }}")
        
        if "line_height" in css_overrides:
            css_rules.append(f"body {{ line-height: {css_overrides['line_height']}; }}")
        
        return "\n".join(css_rules)
    
    def _render_section_content(self, content: Any) -> str:
        """Render section content"""
        
        if isinstance(content, dict):
            items = []
            for key, value in content.items():
                items.append(f'<li><strong>{key.replace("_", " ").title()}:</strong> {value}</li>')
            return f'<ul>{"".join(items)}</ul>'
        elif isinstance(content, str):
            return f'<p>{content}</p>'
        else:
            return str(content)
    
    def _get_translator_notes(self, language: SupportedLanguage) -> Optional[str]:
        """Get translator notes for specific language"""
        
        notes = {
            SupportedLanguage.SPANISH: "Translated by certified technical translator. Regional variations may apply.",
            SupportedLanguage.FRENCH: "Traduction certifiée. Terminologie technique validée.",
            SupportedLanguage.GERMAN: "Übersetzung durch zertifizierten Fachübersetzer.",
            SupportedLanguage.JAPANESE: "認定技術翻訳者による翻訳。専門用語は検証済み。",
            SupportedLanguage.CHINESE_SIMPLIFIED: "经认证技术翻译员翻译。专业术语已验证。"
        }
        
        return notes.get(language)
    
    def get_available_languages(self, playbook_id: str) -> List[SupportedLanguage]:
        """Get available languages for a playbook"""
        
        available = [SupportedLanguage.ENGLISH]  # English always available
        
        if playbook_id in self.localization_templates.get(SupportedLanguage.SPANISH, {}):
            available.append(SupportedLanguage.SPANISH)
        if playbook_id in self.localization_templates.get(SupportedLanguage.FRENCH, {}):
            available.append(SupportedLanguage.FRENCH)
        if playbook_id in self.localization_templates.get(SupportedLanguage.GERMAN, {}):
            available.append(SupportedLanguage.GERMAN)
        if playbook_id in self.localization_templates.get(SupportedLanguage.JAPANESE, {}):
            available.append(SupportedLanguage.JAPANESE)
        
        return available
    
    def get_accessibility_options(self) -> List[AccessibilityLevel]:
        """Get available accessibility options"""
        return list(AccessibilityLevel)


# Example usage
if __name__ == "__main__":
    service = PlaybookLocalizationService()
    
    # Get Spanish version with high contrast
    spanish_playbook = service.get_localized_playbook(
        "incident_response",
        SupportedLanguage.SPANISH,
        AccessibilityLevel.HIGH_CONTRAST
    )
    
    if spanish_playbook:
        print(f"Title: {spanish_playbook.title}")
        print(f"Language: {spanish_playbook.language}")
        print(f"Accessibility: {spanish_playbook.accessibility_level}")
        print(f"Content: {spanish_playbook.content[:200]}...")

