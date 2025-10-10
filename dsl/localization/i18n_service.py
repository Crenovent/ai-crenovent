"""
Task 6.3.80: Internationalization of error messages
Support for multiple languages in error messages
"""

from typing import Dict, Any
from enum import Enum

class SupportedLanguage(Enum):
    """Supported languages for i18n"""
    EN = "en"
    FR = "fr" 
    ES = "es"
    HI = "hi"

class I18nService:
    """
    Internationalization service for error messages
    Task 6.3.80: Global usability with EN/FR/ES/HI support
    """
    
    def __init__(self):
        self.translations = {
            "model_not_found": {
                SupportedLanguage.EN: "Model not found",
                SupportedLanguage.FR: "Modèle non trouvé",
                SupportedLanguage.ES: "Modelo no encontrado", 
                SupportedLanguage.HI: "मॉडल नहीं मिला"
            },
            "inference_timeout": {
                SupportedLanguage.EN: "Inference request timed out",
                SupportedLanguage.FR: "Délai d'attente de la demande d'inférence",
                SupportedLanguage.ES: "Tiempo de espera de solicitud de inferencia",
                SupportedLanguage.HI: "अनुमान अनुरोध का समय समाप्त"
            },
            "insufficient_confidence": {
                SupportedLanguage.EN: "Model confidence below threshold",
                SupportedLanguage.FR: "Confiance du modèle en dessous du seuil",
                SupportedLanguage.ES: "Confianza del modelo por debajo del umbral",
                SupportedLanguage.HI: "मॉडल का विश्वास सीमा से कम"
            }
        }
    
    def get_message(self, message_key: str, language: SupportedLanguage = SupportedLanguage.EN) -> str:
        """Get localized message"""
        if message_key in self.translations:
            return self.translations[message_key].get(language, 
                   self.translations[message_key][SupportedLanguage.EN])
        return message_key

# Global i18n service
i18n_service = I18nService()
