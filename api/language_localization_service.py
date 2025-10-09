"""
Task 3.3.37: Language Localization Service for Conversational Mode
- Multi-region support with SLM fine-tunes
- Residency aware localization
- Support for multiple languages in conversational interactions
"""

from fastapi import FastAPI, HTTPException, BackgroundTasks
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any, Union
from datetime import datetime
from enum import Enum
import uuid
import logging
import json

app = FastAPI(title="RBIA Language Localization Service")
logger = logging.getLogger(__name__)

class SupportedLanguage(str, Enum):
    ENGLISH_US = "en-US"
    ENGLISH_UK = "en-UK"
    SPANISH_ES = "es-ES"
    SPANISH_MX = "es-MX"
    FRENCH_FR = "fr-FR"
    FRENCH_CA = "fr-CA"
    GERMAN_DE = "de-DE"
    ITALIAN_IT = "it-IT"
    PORTUGUESE_BR = "pt-BR"
    PORTUGUESE_PT = "pt-PT"
    DUTCH_NL = "nl-NL"
    JAPANESE_JP = "ja-JP"
    KOREAN_KR = "ko-KR"
    CHINESE_CN = "zh-CN"
    CHINESE_TW = "zh-TW"

class DataResidency(str, Enum):
    US = "us"
    EU = "eu"
    UK = "uk"
    CANADA = "canada"
    ASIA_PACIFIC = "asia_pacific"
    JAPAN = "japan"
    AUSTRALIA = "australia"

class LocalizationScope(str, Enum):
    SYSTEM_MESSAGES = "system_messages"
    USER_PROMPTS = "user_prompts"
    ERROR_MESSAGES = "error_messages"
    HELP_TEXT = "help_text"
    WORKFLOW_DESCRIPTIONS = "workflow_descriptions"
    FIELD_LABELS = "field_labels"
    VALIDATION_MESSAGES = "validation_messages"

class LocalizedContent(BaseModel):
    content_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    tenant_id: str
    
    # Localization details
    language: SupportedLanguage
    residency_region: DataResidency
    scope: LocalizationScope
    
    # Content
    original_key: str  # Key identifier for the content
    original_text: str  # Original English text
    localized_text: str  # Translated text
    context: Optional[str] = None  # Context for better translation
    
    # Metadata
    translator: Optional[str] = None  # Human translator or AI model used
    reviewed_by: Optional[str] = None
    quality_score: Optional[float] = None  # Translation quality score
    
    # Status
    is_active: bool = True
    requires_review: bool = False
    
    created_at: datetime = Field(default_factory=datetime.utcnow)
    last_updated: datetime = Field(default_factory=datetime.utcnow)

class LanguageModel(BaseModel):
    model_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    tenant_id: str
    
    # Model details
    language: SupportedLanguage
    residency_region: DataResidency
    model_name: str
    model_version: str
    
    # Model configuration
    base_model: str  # Base SLM model
    fine_tune_dataset: Optional[str] = None
    fine_tune_status: str = "ready"  # ready, training, failed
    
    # Performance metrics
    accuracy_score: Optional[float] = None
    fluency_score: Optional[float] = None
    coherence_score: Optional[float] = None
    
    # Deployment
    endpoint_url: Optional[str] = None
    api_key: Optional[str] = None
    is_deployed: bool = False
    
    created_at: datetime = Field(default_factory=datetime.utcnow)

class ConversationLocalization(BaseModel):
    localization_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    tenant_id: str
    user_id: str
    session_id: str
    
    # Language settings
    user_language: SupportedLanguage
    detected_language: Optional[SupportedLanguage] = None
    residency_region: DataResidency
    
    # Conversation context
    conversation_history: List[Dict[str, Any]] = Field(default_factory=list)
    localized_responses: List[Dict[str, Any]] = Field(default_factory=list)
    
    # Settings
    auto_detect_language: bool = True
    fallback_to_english: bool = True
    use_regional_model: bool = True
    
    created_at: datetime = Field(default_factory=datetime.utcnow)
    last_interaction: datetime = Field(default_factory=datetime.utcnow)

class TranslationRequest(BaseModel):
    request_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    tenant_id: str
    
    # Translation details
    source_language: SupportedLanguage = SupportedLanguage.ENGLISH_US
    target_language: SupportedLanguage
    residency_region: DataResidency
    
    # Content
    text_to_translate: str
    context: Optional[str] = None
    scope: LocalizationScope = LocalizationScope.SYSTEM_MESSAGES
    
    # Options
    preserve_formatting: bool = True
    preserve_placeholders: bool = True  # Keep {variable} placeholders intact
    use_cached_translation: bool = True
    
    created_at: datetime = Field(default_factory=datetime.utcnow)

class TranslationResponse(BaseModel):
    request_id: str
    translated_text: str
    confidence_score: float
    model_used: str
    cached: bool = False
    processing_time_ms: int
    
    # Quality metrics
    fluency_score: Optional[float] = None
    adequacy_score: Optional[float] = None
    
    created_at: datetime = Field(default_factory=datetime.utcnow)

# In-memory stores (replace with database in production)
localized_content_store: Dict[str, LocalizedContent] = {}
language_models_store: Dict[str, LanguageModel] = {}
conversation_localizations_store: Dict[str, ConversationLocalization] = {}
translation_cache: Dict[str, TranslationResponse] = {}

def _create_default_language_models():
    """Create default language models for supported regions"""
    
    # US English model
    us_model = LanguageModel(
        tenant_id="default",
        language=SupportedLanguage.ENGLISH_US,
        residency_region=DataResidency.US,
        model_name="RBIA-ConversationalAI-EN-US",
        model_version="1.0",
        base_model="microsoft/DialoGPT-medium",
        is_deployed=True,
        accuracy_score=0.92,
        fluency_score=0.95,
        coherence_score=0.90
    )
    
    # EU Spanish model
    eu_spanish_model = LanguageModel(
        tenant_id="default",
        language=SupportedLanguage.SPANISH_ES,
        residency_region=DataResidency.EU,
        model_name="RBIA-ConversationalAI-ES-ES",
        model_version="1.0",
        base_model="microsoft/DialoGPT-medium",
        fine_tune_dataset="spanish_business_conversations",
        is_deployed=True,
        accuracy_score=0.88,
        fluency_score=0.91,
        coherence_score=0.87
    )
    
    # EU French model
    eu_french_model = LanguageModel(
        tenant_id="default",
        language=SupportedLanguage.FRENCH_FR,
        residency_region=DataResidency.EU,
        model_name="RBIA-ConversationalAI-FR-FR",
        model_version="1.0",
        base_model="microsoft/DialoGPT-medium",
        fine_tune_dataset="french_business_conversations",
        is_deployed=True,
        accuracy_score=0.89,
        fluency_score=0.92,
        coherence_score=0.88
    )
    
    # Asia Pacific Japanese model
    jp_model = LanguageModel(
        tenant_id="default",
        language=SupportedLanguage.JAPANESE_JP,
        residency_region=DataResidency.JAPAN,
        model_name="RBIA-ConversationalAI-JA-JP",
        model_version="1.0",
        base_model="rinna/japanese-gpt2-medium",
        fine_tune_dataset="japanese_business_conversations",
        is_deployed=True,
        accuracy_score=0.85,
        fluency_score=0.89,
        coherence_score=0.84
    )
    
    # Store default models
    language_models_store[us_model.model_id] = us_model
    language_models_store[eu_spanish_model.model_id] = eu_spanish_model
    language_models_store[eu_french_model.model_id] = eu_french_model
    language_models_store[jp_model.model_id] = jp_model
    
    logger.info("Created default language models")

def _create_default_localized_content():
    """Create default localized content for common system messages"""
    
    # System greeting messages
    greetings = [
        {
            "key": "system.greeting.welcome",
            "original": "Welcome to RBIA! How can I help you today?",
            "translations": {
                SupportedLanguage.SPANISH_ES: "¡Bienvenido a RBIA! ¿Cómo puedo ayudarte hoy?",
                SupportedLanguage.FRENCH_FR: "Bienvenue dans RBIA ! Comment puis-je vous aider aujourd'hui ?",
                SupportedLanguage.GERMAN_DE: "Willkommen bei RBIA! Wie kann ich Ihnen heute helfen?",
                SupportedLanguage.JAPANESE_JP: "RBIAへようこそ！今日はどのようにお手伝いできますか？"
            }
        },
        {
            "key": "system.error.general",
            "original": "I'm sorry, I encountered an error. Please try again.",
            "translations": {
                SupportedLanguage.SPANISH_ES: "Lo siento, encontré un error. Por favor, inténtalo de nuevo.",
                SupportedLanguage.FRENCH_FR: "Je suis désolé, j'ai rencontré une erreur. Veuillez réessayer.",
                SupportedLanguage.GERMAN_DE: "Es tut mir leid, ich bin auf einen Fehler gestoßen. Bitte versuchen Sie es erneut.",
                SupportedLanguage.JAPANESE_JP: "申し訳ございませんが、エラーが発生しました。もう一度お試しください。"
            }
        }
    ]
    
    for greeting_data in greetings:
        for language, translation in greeting_data["translations"].items():
            # Determine residency based on language
            residency = _get_residency_for_language(language)
            
            content = LocalizedContent(
                tenant_id="default",
                language=language,
                residency_region=residency,
                scope=LocalizationScope.SYSTEM_MESSAGES,
                original_key=greeting_data["key"],
                original_text=greeting_data["original"],
                localized_text=translation,
                translator="AI_INITIAL",
                quality_score=0.85
            )
            
            localized_content_store[content.content_id] = content
    
    logger.info("Created default localized content")

def _get_residency_for_language(language: SupportedLanguage) -> DataResidency:
    """Determine data residency region based on language"""
    
    language_to_residency = {
        SupportedLanguage.ENGLISH_US: DataResidency.US,
        SupportedLanguage.ENGLISH_UK: DataResidency.UK,
        SupportedLanguage.SPANISH_ES: DataResidency.EU,
        SupportedLanguage.SPANISH_MX: DataResidency.US,
        SupportedLanguage.FRENCH_FR: DataResidency.EU,
        SupportedLanguage.FRENCH_CA: DataResidency.CANADA,
        SupportedLanguage.GERMAN_DE: DataResidency.EU,
        SupportedLanguage.ITALIAN_IT: DataResidency.EU,
        SupportedLanguage.PORTUGUESE_BR: DataResidency.US,
        SupportedLanguage.PORTUGUESE_PT: DataResidency.EU,
        SupportedLanguage.DUTCH_NL: DataResidency.EU,
        SupportedLanguage.JAPANESE_JP: DataResidency.JAPAN,
        SupportedLanguage.KOREAN_KR: DataResidency.ASIA_PACIFIC,
        SupportedLanguage.CHINESE_CN: DataResidency.ASIA_PACIFIC,
        SupportedLanguage.CHINESE_TW: DataResidency.ASIA_PACIFIC,
    }
    
    return language_to_residency.get(language, DataResidency.US)

# Initialize default data
_create_default_language_models()
_create_default_localized_content()

@app.post("/localization/models", response_model=LanguageModel)
async def create_language_model(model: LanguageModel):
    """Create or register a new language model"""
    language_models_store[model.model_id] = model
    logger.info(f"Created language model for {model.language.value} in {model.residency_region.value}")
    return model

@app.get("/localization/models", response_model=List[LanguageModel])
async def get_language_models(
    tenant_id: str,
    language: Optional[SupportedLanguage] = None,
    residency_region: Optional[DataResidency] = None
):
    """Get available language models"""
    models = [m for m in language_models_store.values() if m.tenant_id == tenant_id]
    
    if language:
        models = [m for m in models if m.language == language]
    
    if residency_region:
        models = [m for m in models if m.residency_region == residency_region]
    
    return models

@app.post("/localization/conversations", response_model=ConversationLocalization)
async def create_conversation_localization(
    tenant_id: str,
    user_id: str,
    session_id: str,
    user_language: SupportedLanguage,
    residency_region: Optional[DataResidency] = None
):
    """Create a new localized conversation session"""
    
    # Auto-determine residency if not provided
    if not residency_region:
        residency_region = _get_residency_for_language(user_language)
    
    localization = ConversationLocalization(
        tenant_id=tenant_id,
        user_id=user_id,
        session_id=session_id,
        user_language=user_language,
        residency_region=residency_region
    )
    
    conversation_localizations_store[localization.localization_id] = localization
    logger.info(f"Created conversation localization for {user_language.value} in {residency_region.value}")
    return localization

@app.post("/localization/translate", response_model=TranslationResponse)
async def translate_text(
    request: TranslationRequest,
    background_tasks: BackgroundTasks
):
    """Translate text to target language with residency awareness"""
    
    # Check cache first
    cache_key = f"{request.source_language.value}_{request.target_language.value}_{hash(request.text_to_translate)}"
    
    if request.use_cached_translation and cache_key in translation_cache:
        cached_response = translation_cache[cache_key]
        logger.info(f"Returning cached translation for request {request.request_id}")
        return cached_response
    
    # Find appropriate model for target language and residency
    suitable_models = [
        model for model in language_models_store.values()
        if (model.language == request.target_language and
            model.residency_region == request.residency_region and
            model.is_deployed)
    ]
    
    if not suitable_models:
        # Fallback to any model with the target language
        suitable_models = [
            model for model in language_models_store.values()
            if model.language == request.target_language and model.is_deployed
        ]
    
    if not suitable_models:
        raise HTTPException(
            status_code=404, 
            detail=f"No suitable model found for {request.target_language.value} in {request.residency_region.value}"
        )
    
    # Use the best model (highest accuracy)
    best_model = max(suitable_models, key=lambda m: m.accuracy_score or 0)
    
    # Simulate translation (in production, this would call the actual model)
    translated_text = _simulate_translation(
        request.text_to_translate, 
        request.source_language, 
        request.target_language,
        request.preserve_placeholders
    )
    
    response = TranslationResponse(
        request_id=request.request_id,
        translated_text=translated_text,
        confidence_score=best_model.accuracy_score or 0.85,
        model_used=best_model.model_name,
        processing_time_ms=150,  # Simulated processing time
        fluency_score=best_model.fluency_score,
        adequacy_score=best_model.coherence_score
    )
    
    # Cache the response
    translation_cache[cache_key] = response
    
    logger.info(f"Translated text using {best_model.model_name}")
    return response

def _simulate_translation(
    text: str, 
    source_lang: SupportedLanguage, 
    target_lang: SupportedLanguage,
    preserve_placeholders: bool = True
) -> str:
    """Simulate translation (replace with actual translation service in production)"""
    
    # Simple simulation - in production, this would call the actual SLM
    translations = {
        ("en-US", "es-ES"): {
            "Welcome to RBIA! How can I help you today?": "¡Bienvenido a RBIA! ¿Cómo puedo ayudarte hoy?",
            "I'm sorry, I encountered an error. Please try again.": "Lo siento, encontré un error. Por favor, inténtalo de nuevo.",
            "Your workflow has been created successfully.": "Tu flujo de trabajo se ha creado exitosamente.",
            "The model confidence is {confidence:.2f}": "La confianza del modelo es {confidence:.2f}"
        },
        ("en-US", "fr-FR"): {
            "Welcome to RBIA! How can I help you today?": "Bienvenue dans RBIA ! Comment puis-je vous aider aujourd'hui ?",
            "I'm sorry, I encountered an error. Please try again.": "Je suis désolé, j'ai rencontré une erreur. Veuillez réessayer.",
            "Your workflow has been created successfully.": "Votre flux de travail a été créé avec succès.",
            "The model confidence is {confidence:.2f}": "La confiance du modèle est {confidence:.2f}"
        },
        ("en-US", "ja-JP"): {
            "Welcome to RBIA! How can I help you today?": "RBIAへようこそ！今日はどのようにお手伝いできますか？",
            "I'm sorry, I encountered an error. Please try again.": "申し訳ございませんが、エラーが発生しました。もう一度お試しください。",
            "Your workflow has been created successfully.": "ワークフローが正常に作成されました。",
            "The model confidence is {confidence:.2f}": "モデルの信頼度は{confidence:.2f}です"
        }
    }
    
    lang_pair = (source_lang.value, target_lang.value)
    translation_dict = translations.get(lang_pair, {})
    
    # Return translation if found, otherwise return original text with a prefix
    translated = translation_dict.get(text, f"[{target_lang.value}] {text}")
    
    return translated

@app.get("/localization/content", response_model=List[LocalizedContent])
async def get_localized_content(
    tenant_id: str,
    language: Optional[SupportedLanguage] = None,
    scope: Optional[LocalizationScope] = None,
    residency_region: Optional[DataResidency] = None
):
    """Get localized content for a tenant"""
    
    content = [c for c in localized_content_store.values() if c.tenant_id == tenant_id and c.is_active]
    
    if language:
        content = [c for c in content if c.language == language]
    
    if scope:
        content = [c for c in content if c.scope == scope]
    
    if residency_region:
        content = [c for c in content if c.residency_region == residency_region]
    
    return content

@app.post("/localization/content", response_model=LocalizedContent)
async def create_localized_content(content: LocalizedContent):
    """Create new localized content"""
    localized_content_store[content.content_id] = content
    logger.info(f"Created localized content for {content.language.value}: {content.original_key}")
    return content

@app.get("/localization/conversations/{session_id}/history")
async def get_localized_conversation_history(session_id: str):
    """Get localized conversation history"""
    
    localization = None
    for loc in conversation_localizations_store.values():
        if loc.session_id == session_id:
            localization = loc
            break
    
    if not localization:
        raise HTTPException(status_code=404, detail="Conversation localization not found")
    
    return {
        "session_id": session_id,
        "user_language": localization.user_language.value,
        "residency_region": localization.residency_region.value,
        "conversation_history": localization.conversation_history,
        "localized_responses": localization.localized_responses
    }

@app.post("/localization/conversations/{session_id}/message")
async def add_localized_message(
    session_id: str,
    message: str,
    is_user_message: bool = True,
    background_tasks: BackgroundTasks = BackgroundTasks()
):
    """Add a message to localized conversation"""
    
    localization = None
    for loc in conversation_localizations_store.values():
        if loc.session_id == session_id:
            localization = loc
            break
    
    if not localization:
        raise HTTPException(status_code=404, detail="Conversation localization not found")
    
    # Add to conversation history
    message_entry = {
        "timestamp": datetime.utcnow().isoformat(),
        "message": message,
        "is_user_message": is_user_message,
        "language": localization.user_language.value
    }
    
    localization.conversation_history.append(message_entry)
    localization.last_interaction = datetime.utcnow()
    
    # If it's a system message, ensure it's properly localized
    if not is_user_message:
        background_tasks.add_task(_ensure_message_localization, localization, message)
    
    logger.info(f"Added message to localized conversation {session_id}")
    return {"status": "added", "message_id": len(localization.conversation_history)}

async def _ensure_message_localization(localization: ConversationLocalization, message: str):
    """Ensure system message is properly localized"""
    
    # Check if we have a localized version of this message
    localized_content = [
        content for content in localized_content_store.values()
        if (content.tenant_id == localization.tenant_id and
            content.language == localization.user_language and
            content.original_text == message)
    ]
    
    if not localized_content and localization.user_language != SupportedLanguage.ENGLISH_US:
        # Need to translate this message
        translation_request = TranslationRequest(
            tenant_id=localization.tenant_id,
            target_language=localization.user_language,
            residency_region=localization.residency_region,
            text_to_translate=message
        )
        
        # In production, this would be an async call to the translation service
        logger.info(f"Would translate message for session {localization.session_id}")

@app.get("/localization/analytics/usage")
async def get_localization_usage_analytics(tenant_id: str):
    """Get analytics on localization usage"""
    
    tenant_localizations = [
        loc for loc in conversation_localizations_store.values()
        if loc.tenant_id == tenant_id
    ]
    
    analytics = {
        "total_conversations": len(tenant_localizations),
        "by_language": {},
        "by_residency_region": {},
        "most_active_language": None,
        "translation_requests": 0
    }
    
    # Analyze by language
    for language in SupportedLanguage:
        lang_conversations = [loc for loc in tenant_localizations if loc.user_language == language]
        analytics["by_language"][language.value] = len(lang_conversations)
    
    # Analyze by residency region
    for residency in DataResidency:
        region_conversations = [loc for loc in tenant_localizations if loc.residency_region == residency]
        analytics["by_residency_region"][residency.value] = len(region_conversations)
    
    # Find most active language
    if analytics["by_language"]:
        most_active = max(analytics["by_language"].items(), key=lambda x: x[1])
        analytics["most_active_language"] = most_active[0]
    
    # Count translation requests (from cache)
    analytics["translation_requests"] = len(translation_cache)
    
    return analytics

@app.get("/health")
async def health_check():
    return {"status": "healthy", "service": "RBIA Language Localization Service", "task": "3.3.37"}
