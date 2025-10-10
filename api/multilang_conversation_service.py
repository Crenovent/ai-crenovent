"""
Task 3.4.46: Provide multi-language Conversational UX support
- SLM fine-tuning for multiple languages
- Language detection and routing
- Multi-language prompt templates
- Conversation context management across languages
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from datetime import datetime
from enum import Enum
import uuid
import re

app = FastAPI(title="RBIA Multi-Language Conversational UX")

class SupportedLanguage(str, Enum):
    ENGLISH = "en"
    SPANISH = "es"
    FRENCH = "fr"
    GERMAN = "de"
    ITALIAN = "it"
    PORTUGUESE = "pt"
    DUTCH = "nl"
    JAPANESE = "ja"
    KOREAN = "ko"
    CHINESE_SIMPLIFIED = "zh-cn"
    CHINESE_TRADITIONAL = "zh-tw"
    HINDI = "hi"
    ARABIC = "ar"

class ConversationContext(BaseModel):
    context_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    tenant_id: str
    user_id: str
    
    # Language settings
    primary_language: SupportedLanguage
    detected_language: Optional[SupportedLanguage] = None
    language_confidence: float = 1.0
    
    # Conversation state
    conversation_history: List[Dict[str, Any]] = Field(default_factory=list)
    current_intent: Optional[str] = None
    workflow_context: Dict[str, Any] = Field(default_factory=dict)
    
    # Translation settings
    auto_translate: bool = True
    preserve_technical_terms: bool = True
    
    created_at: datetime = Field(default_factory=datetime.utcnow)
    last_interaction: datetime = Field(default_factory=datetime.utcnow)

class MultiLangPromptTemplate(BaseModel):
    template_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    template_name: str
    category: str  # workflow_guidance, error_explanation, help_text
    
    # Localized templates
    templates: Dict[SupportedLanguage, str] = Field(default_factory=dict)
    
    # Template variables
    variables: List[str] = Field(default_factory=list)
    
    # Metadata
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)

class ConversationMessage(BaseModel):
    message_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    context_id: str
    
    # Message content
    original_text: str
    detected_language: SupportedLanguage
    translated_text: Optional[str] = None
    target_language: Optional[SupportedLanguage] = None
    
    # Processing
    intent: Optional[str] = None
    entities: Dict[str, Any] = Field(default_factory=dict)
    confidence: float = 0.0
    
    # Response
    response_text: str
    response_language: SupportedLanguage
    
    timestamp: datetime = Field(default_factory=datetime.utcnow)

# Storage
contexts_store: Dict[str, ConversationContext] = {}
templates_store: Dict[str, MultiLangPromptTemplate] = {}
messages_store: Dict[str, ConversationMessage] = {}

def _initialize_default_templates():
    """Initialize default multi-language templates"""
    
    # Workflow guidance template
    workflow_template = MultiLangPromptTemplate(
        template_id="workflow_guidance",
        template_name="Workflow Guidance",
        category="workflow_guidance",
        templates={
            SupportedLanguage.ENGLISH: "I'll help you create a {workflow_type} workflow. What would you like to accomplish?",
            SupportedLanguage.SPANISH: "Te ayudaré a crear un flujo de trabajo de {workflow_type}. ¿Qué te gustaría lograr?",
            SupportedLanguage.FRENCH: "Je vais vous aider à créer un workflow {workflow_type}. Que souhaitez-vous accomplir?",
            SupportedLanguage.GERMAN: "Ich helfe Ihnen dabei, einen {workflow_type}-Workflow zu erstellen. Was möchten Sie erreichen?",
            SupportedLanguage.CHINESE_SIMPLIFIED: "我将帮助您创建{workflow_type}工作流。您想要完成什么？"
        },
        variables=["workflow_type"]
    )
    
    templates_store[workflow_template.template_id] = workflow_template
    
    # Error explanation template
    error_template = MultiLangPromptTemplate(
        template_id="error_explanation",
        template_name="Error Explanation",
        category="error_explanation",
        templates={
            SupportedLanguage.ENGLISH: "I encountered an error: {error_message}. Let me help you resolve this.",
            SupportedLanguage.SPANISH: "Encontré un error: {error_message}. Permíteme ayudarte a resolverlo.",
            SupportedLanguage.FRENCH: "J'ai rencontré une erreur: {error_message}. Laissez-moi vous aider à la résoudre.",
            SupportedLanguage.GERMAN: "Ich bin auf einen Fehler gestoßen: {error_message}. Lassen Sie mich Ihnen bei der Lösung helfen.",
            SupportedLanguage.CHINESE_SIMPLIFIED: "我遇到了一个错误：{error_message}。让我帮您解决这个问题。"
        },
        variables=["error_message"]
    )
    
    templates_store[error_template.template_id] = error_template

@app.on_event("startup")
async def startup_event():
    _initialize_default_templates()

@app.post("/conversation/start", response_model=ConversationContext)
async def start_conversation(
    tenant_id: str,
    user_id: str,
    primary_language: SupportedLanguage,
    initial_message: Optional[str] = None
):
    """Start a new multi-language conversation"""
    
    context = ConversationContext(
        tenant_id=tenant_id,
        user_id=user_id,
        primary_language=primary_language
    )
    
    if initial_message:
        # Detect language of initial message
        detected_lang, confidence = _detect_language(initial_message)
        context.detected_language = detected_lang
        context.language_confidence = confidence
        
        # Add to conversation history
        context.conversation_history.append({
            "role": "user",
            "content": initial_message,
            "language": detected_lang.value,
            "timestamp": datetime.utcnow().isoformat()
        })
    
    contexts_store[context.context_id] = context
    return context

@app.post("/conversation/message", response_model=ConversationMessage)
async def process_message(
    context_id: str,
    message_text: str,
    target_language: Optional[SupportedLanguage] = None
):
    """Process a conversational message with multi-language support"""
    
    if context_id not in contexts_store:
        raise HTTPException(status_code=404, detail="Conversation context not found")
    
    context = contexts_store[context_id]
    
    # Detect language
    detected_lang, confidence = _detect_language(message_text)
    
    # Translate if needed
    translated_text = None
    if target_language and detected_lang != target_language:
        translated_text = _translate_text(message_text, detected_lang, target_language)
    
    # Process intent and entities
    intent, entities = _process_intent(translated_text or message_text, target_language or detected_lang)
    
    # Generate response
    response_text = _generate_response(intent, entities, context, target_language or detected_lang)
    
    # Create message record
    message = ConversationMessage(
        context_id=context_id,
        original_text=message_text,
        detected_language=detected_lang,
        translated_text=translated_text,
        target_language=target_language,
        intent=intent,
        entities=entities,
        confidence=confidence,
        response_text=response_text,
        response_language=target_language or detected_lang
    )
    
    # Update conversation context
    context.conversation_history.append({
        "role": "user",
        "content": message_text,
        "language": detected_lang.value,
        "timestamp": datetime.utcnow().isoformat()
    })
    
    context.conversation_history.append({
        "role": "assistant",
        "content": response_text,
        "language": (target_language or detected_lang).value,
        "timestamp": datetime.utcnow().isoformat()
    })
    
    context.current_intent = intent
    context.last_interaction = datetime.utcnow()
    
    contexts_store[context_id] = context
    messages_store[message.message_id] = message
    
    return message

@app.get("/conversation/context/{context_id}", response_model=ConversationContext)
async def get_conversation_context(context_id: str):
    """Get conversation context"""
    
    if context_id not in contexts_store:
        raise HTTPException(status_code=404, detail="Conversation context not found")
    
    return contexts_store[context_id]

@app.post("/conversation/switch-language")
async def switch_conversation_language(
    context_id: str,
    new_language: SupportedLanguage
):
    """Switch conversation language"""
    
    if context_id not in contexts_store:
        raise HTTPException(status_code=404, detail="Conversation context not found")
    
    context = contexts_store[context_id]
    old_language = context.primary_language
    context.primary_language = new_language
    
    # Add language switch notification to history
    switch_message = _get_template_text("language_switch", new_language, {
        "old_language": old_language.value,
        "new_language": new_language.value
    })
    
    context.conversation_history.append({
        "role": "system",
        "content": switch_message,
        "language": new_language.value,
        "timestamp": datetime.utcnow().isoformat()
    })
    
    contexts_store[context_id] = context
    
    return {
        "context_id": context_id,
        "old_language": old_language.value,
        "new_language": new_language.value,
        "message": switch_message
    }

@app.get("/templates", response_model=List[MultiLangPromptTemplate])
async def list_templates(category: Optional[str] = None):
    """List available multi-language templates"""
    
    templates = list(templates_store.values())
    
    if category:
        templates = [t for t in templates if t.category == category]
    
    return templates

@app.post("/templates", response_model=MultiLangPromptTemplate)
async def create_template(template: MultiLangPromptTemplate):
    """Create new multi-language template"""
    
    templates_store[template.template_id] = template
    return template

@app.get("/languages/supported", response_model=List[Dict[str, str]])
async def get_supported_languages():
    """Get list of supported languages"""
    
    language_info = [
        {"code": lang.value, "name": _get_language_name(lang), "native_name": _get_native_name(lang)}
        for lang in SupportedLanguage
    ]
    
    return language_info

@app.get("/conversation/analytics/{tenant_id}")
async def get_language_analytics(tenant_id: str):
    """Get language usage analytics for tenant"""
    
    # Filter contexts by tenant
    tenant_contexts = [c for c in contexts_store.values() if c.tenant_id == tenant_id]
    
    if not tenant_contexts:
        return {"message": "No conversation data found for tenant"}
    
    # Calculate language distribution
    language_usage = {}
    for context in tenant_contexts:
        lang = context.primary_language.value
        language_usage[lang] = language_usage.get(lang, 0) + 1
    
    # Calculate conversation metrics
    total_conversations = len(tenant_contexts)
    avg_messages_per_conversation = sum(len(c.conversation_history) for c in tenant_contexts) / total_conversations
    
    # Language switching patterns
    language_switches = sum(1 for c in tenant_contexts for msg in c.conversation_history if msg.get("role") == "system" and "language" in msg.get("content", ""))
    
    return {
        "tenant_id": tenant_id,
        "total_conversations": total_conversations,
        "language_distribution": language_usage,
        "average_messages_per_conversation": avg_messages_per_conversation,
        "language_switches": language_switches,
        "most_popular_language": max(language_usage, key=language_usage.get) if language_usage else "en",
        "multilingual_users": len([c for c in tenant_contexts if c.detected_language != c.primary_language])
    }

def _detect_language(text: str) -> tuple[SupportedLanguage, float]:
    """Detect language of input text (simplified implementation)"""
    
    # Simple language detection based on common words/patterns
    text_lower = text.lower()
    
    # Spanish indicators
    if any(word in text_lower for word in ["hola", "gracias", "por favor", "sí", "no", "cómo"]):
        return SupportedLanguage.SPANISH, 0.8
    
    # French indicators
    if any(word in text_lower for word in ["bonjour", "merci", "s'il vous plaît", "oui", "non", "comment"]):
        return SupportedLanguage.FRENCH, 0.8
    
    # German indicators
    if any(word in text_lower for word in ["hallo", "danke", "bitte", "ja", "nein", "wie"]):
        return SupportedLanguage.GERMAN, 0.8
    
    # Chinese indicators (simplified)
    if re.search(r'[\u4e00-\u9fff]', text):
        return SupportedLanguage.CHINESE_SIMPLIFIED, 0.9
    
    # Default to English
    return SupportedLanguage.ENGLISH, 0.7

def _translate_text(text: str, source_lang: SupportedLanguage, target_lang: SupportedLanguage) -> str:
    """Translate text between languages (simplified implementation)"""
    
    # Simple translation mapping for demo
    translations = {
        ("en", "es"): {"hello": "hola", "thank you": "gracias", "yes": "sí", "no": "no"},
        ("en", "fr"): {"hello": "bonjour", "thank you": "merci", "yes": "oui", "no": "non"},
        ("en", "de"): {"hello": "hallo", "thank you": "danke", "yes": "ja", "no": "nein"},
    }
    
    translation_key = (source_lang.value, target_lang.value)
    if translation_key in translations:
        translated = text.lower()
        for eng, foreign in translations[translation_key].items():
            translated = translated.replace(eng, foreign)
        return translated
    
    # Fallback: return original text with language note
    return f"[{target_lang.value}] {text}"

def _process_intent(text: str, language: SupportedLanguage) -> tuple[Optional[str], Dict[str, Any]]:
    """Process intent and extract entities from text"""
    
    text_lower = text.lower()
    
    # Simple intent detection
    if any(word in text_lower for word in ["create", "make", "build", "new", "crear", "faire", "erstellen"]):
        return "create_workflow", {"action": "create", "object": "workflow"}
    
    if any(word in text_lower for word in ["help", "assist", "support", "ayuda", "aide", "hilfe"]):
        return "request_help", {"topic": "general"}
    
    if any(word in text_lower for word in ["error", "problem", "issue", "error", "problème", "fehler"]):
        return "report_error", {"severity": "unknown"}
    
    return "general_query", {}

def _generate_response(intent: Optional[str], entities: Dict[str, Any], context: ConversationContext, language: SupportedLanguage) -> str:
    """Generate response based on intent and context"""
    
    # Get appropriate template
    if intent == "create_workflow":
        return _get_template_text("workflow_guidance", language, {"workflow_type": "new"})
    elif intent == "request_help":
        return _get_template_text("help_response", language, {})
    elif intent == "report_error":
        return _get_template_text("error_explanation", language, {"error_message": "general error"})
    else:
        return _get_template_text("general_response", language, {})

def _get_template_text(template_name: str, language: SupportedLanguage, variables: Dict[str, str]) -> str:
    """Get localized template text"""
    
    # Default responses if template not found
    default_responses = {
        "workflow_guidance": {
            SupportedLanguage.ENGLISH: "I'll help you create a workflow. What would you like to accomplish?",
            SupportedLanguage.SPANISH: "Te ayudaré a crear un flujo de trabajo. ¿Qué te gustaría lograr?",
            SupportedLanguage.FRENCH: "Je vais vous aider à créer un workflow. Que souhaitez-vous accomplir?",
            SupportedLanguage.GERMAN: "Ich helfe Ihnen dabei, einen Workflow zu erstellen. Was möchten Sie erreichen?",
        },
        "help_response": {
            SupportedLanguage.ENGLISH: "I'm here to help! What do you need assistance with?",
            SupportedLanguage.SPANISH: "¡Estoy aquí para ayudar! ¿Con qué necesitas asistencia?",
            SupportedLanguage.FRENCH: "Je suis là pour vous aider! De quoi avez-vous besoin?",
            SupportedLanguage.GERMAN: "Ich bin hier, um zu helfen! Wobei brauchen Sie Unterstützung?",
        },
        "general_response": {
            SupportedLanguage.ENGLISH: "I understand. How can I help you further?",
            SupportedLanguage.SPANISH: "Entiendo. ¿Cómo puedo ayudarte más?",
            SupportedLanguage.FRENCH: "Je comprends. Comment puis-je vous aider davantage?",
            SupportedLanguage.GERMAN: "Ich verstehe. Wie kann ich Ihnen weiterhelfen?",
        }
    }
    
    # Get template text
    template_responses = default_responses.get(template_name, {})
    response = template_responses.get(language, template_responses.get(SupportedLanguage.ENGLISH, "I'm here to help!"))
    
    # Apply variables
    for var, value in variables.items():
        response = response.replace(f"{{{var}}}", value)
    
    return response

def _get_language_name(language: SupportedLanguage) -> str:
    """Get English name of language"""
    
    names = {
        SupportedLanguage.ENGLISH: "English",
        SupportedLanguage.SPANISH: "Spanish",
        SupportedLanguage.FRENCH: "French",
        SupportedLanguage.GERMAN: "German",
        SupportedLanguage.ITALIAN: "Italian",
        SupportedLanguage.PORTUGUESE: "Portuguese",
        SupportedLanguage.DUTCH: "Dutch",
        SupportedLanguage.JAPANESE: "Japanese",
        SupportedLanguage.KOREAN: "Korean",
        SupportedLanguage.CHINESE_SIMPLIFIED: "Chinese (Simplified)",
        SupportedLanguage.CHINESE_TRADITIONAL: "Chinese (Traditional)",
        SupportedLanguage.HINDI: "Hindi",
        SupportedLanguage.ARABIC: "Arabic"
    }
    
    return names.get(language, language.value)

def _get_native_name(language: SupportedLanguage) -> str:
    """Get native name of language"""
    
    native_names = {
        SupportedLanguage.ENGLISH: "English",
        SupportedLanguage.SPANISH: "Español",
        SupportedLanguage.FRENCH: "Français",
        SupportedLanguage.GERMAN: "Deutsch",
        SupportedLanguage.ITALIAN: "Italiano",
        SupportedLanguage.PORTUGUESE: "Português",
        SupportedLanguage.DUTCH: "Nederlands",
        SupportedLanguage.JAPANESE: "日本語",
        SupportedLanguage.KOREAN: "한국어",
        SupportedLanguage.CHINESE_SIMPLIFIED: "简体中文",
        SupportedLanguage.CHINESE_TRADITIONAL: "繁體中文",
        SupportedLanguage.HINDI: "हिन्दी",
        SupportedLanguage.ARABIC: "العربية"
    }
    
    return native_names.get(language, language.value)

@app.get("/health")
async def health_check():
    return {"status": "healthy", "service": "Multi-Language Conversational UX", "task": "3.4.46"}
