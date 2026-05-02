"""
language.py — ShopNow AI Voice Agent
Language detection and multilingual support for English, Hindi, and Tamil.
Handles mid-conversation code-switching by tracking per-turn language history.
"""

from __future__ import annotations

import re


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

SUPPORTED_LANGUAGES: dict[str, str] = {
    "en": "English",
    "hi": "Hindi",
    "ta": "Tamil",
}

# Romanised Hindi clue-words (common transliterations)
_ROMANISED_HINDI_CLUES = [
    "aapka", "aapki", "mera", "meri", "kya", "kab", "kahan",
    "kaisa", "kaisi", "nahi", "nahin", "haan", "theek", "abhi",
    "order", "karo", "karo", "chahiye", "milega", "gaya",
]

# Romanised Tamil clue-words
_ROMANISED_TAMIL_CLUES = [
    "enna", "eppo", "enga", "sollunga", "vandhuchu", "vandha",
    "illai", "aam", "romba", "konjam", "theriyuma",
]

# Pre-canned response templates in all three languages
_TEMPLATES: dict[str, dict[str, str]] = {
    "greeting": {
        "en": "Hello! Welcome to ShopNow Support. How can I help you today?",
        "hi": "नमस्ते! ShopNow सपोर्ट में आपका स्वागत है। आज मैं आपकी कैसे सहायता कर सकता हूँ?",
        "ta": "வணக்கம்! ShopNow ஆதரவுக்கு வரவேற்கிறோம். இன்று நான் உங்களுக்கு எப்படி உதவலாம்?",
    },
    "escalation_notice": {
        "en": (
            "I completely understand your frustration and I'm sincerely sorry for the "
            "inconvenience. Let me connect you with a senior specialist who will personally "
            "ensure this is resolved for you."
        ),
        "hi": (
            "मैं आपकी परेशानी को पूरी तरह समझता हूँ और हुई असुविधा के लिए मैं ईमानदारी से "
            "माफी माँगता हूँ। मैं आपको एक वरिष्ठ विशेषज्ञ से जोड़ रहा हूँ जो इसका समाधान "
            "सुनिश्चित करेंगे।"
        ),
        "ta": (
            "உங்கள் கோபத்தை நான் புரிந்துகொள்கிறேன், மேலும் இதற்காக மனசாரை மன்னிப்பு "
            "கேட்கிறேன். ஒரு மூத்த நிபுணரிடம் உங்களை இணைக்கிறேன், அவர் இதனை தீர்த்து "
            "வைப்பார்."
        ),
    },
    "apology": {
        "en": "I sincerely apologise for the inconvenience caused.",
        "hi": "हुई असुविधा के लिए मैं ईमानदारी से माफी माँगता हूँ।",
        "ta": "ஏற்பட்ட தொந்தரவுக்கு நான் மனசாரை மன்னிப்பு கேட்கிறேன்.",
    },
    "resolution_confirm": {
        "en": "Is there anything else I can help you with today?",
        "hi": "क्या आज मैं आपकी किसी और तरह से सहायता कर सकता हूँ?",
        "ta": "இன்று வேறு ஏதாவது உதவி தேவைப்படுகிறதா?",
    },
}


# ---------------------------------------------------------------------------
# Language Detector
# ---------------------------------------------------------------------------

class LanguageDetector:
    """
    Detects the language of each message using Unicode script analysis and
    romanised-keyword matching.  Supported languages:
      • en — English
      • hi — Hindi (Devanagari script: U+0900–U+097F)
      • ta — Tamil  (Tamil script: U+0B80–U+0BFF)

    Tracks a per-session language history to detect code-switching.
    """

    def __init__(self) -> None:
        self._history: list[str] = []

    # ------------------------------------------------------------------
    # Detection
    # ------------------------------------------------------------------

    def detect(self, text: str) -> str:
        """
        Detect the primary language of `text`.
        Returns an ISO-639-1 code: 'en', 'hi', or 'ta'.
        Updates internal history for code-switching detection.
        """
        hindi_chars = len(re.findall(r"[\u0900-\u097F]", text))
        tamil_chars = len(re.findall(r"[\u0B80-\u0BFF]", text))
        total_chars = max(len(text.replace(" ", "")), 1)

        if hindi_chars >= 3 or (hindi_chars > 0 and hindi_chars / total_chars > 0.20):
            detected = "hi"
        elif tamil_chars >= 3 or (tamil_chars > 0 and tamil_chars / total_chars > 0.20):
            detected = "ta"
        else:
            # Romanised keyword fallback
            lower = text.lower()
            hindi_score = sum(1 for kw in _ROMANISED_HINDI_CLUES if kw in lower)
            tamil_score = sum(1 for kw in _ROMANISED_TAMIL_CLUES if kw in lower)
            if hindi_score >= 2:
                detected = "hi"
            elif tamil_score >= 2:
                detected = "ta"
            else:
                detected = "en"

        self._history.append(detected)
        return detected

    # ------------------------------------------------------------------
    # Code-switching
    # ------------------------------------------------------------------

    def detect_code_switching(self) -> bool:
        """Return True if the current turn uses a different language than the previous one."""
        return len(self._history) >= 2 and self._history[-1] != self._history[-2]

    def get_dominant_language(self) -> str:
        """Return the most-used language code in this session."""
        if not self._history:
            return "en"
        return max(set(self._history), key=self._history.count)

    @property
    def current_language(self) -> str:
        """Most recently detected language code."""
        return self._history[-1] if self._history else "en"

    # ------------------------------------------------------------------
    # Prompt-engineering helpers
    # ------------------------------------------------------------------

    @staticmethod
    def get_language_instruction(lang_code: str) -> str:
        """
        Return a short instruction string appended to the system prompt so the
        LLM responds in the correct language.
        """
        instructions = {
            "en": "Always respond in clear, professional English.",
            "hi": (
                "हमेशा सरल और स्पष्ट हिंदी में जवाब दें। "
                "(Always respond in Hindi using simple, clear language.)"
            ),
            "ta": (
                "எப்போதும் தெளிவான தமிழில் பதிலளிக்கவும். "
                "(Always respond in Tamil using simple, clear language.)"
            ),
        }
        return instructions.get(lang_code, instructions["en"])

    # ------------------------------------------------------------------
    # Utility helpers
    # ------------------------------------------------------------------

    @staticmethod
    def get_language_name(code: str) -> str:
        """Convert an ISO language code to a human-readable name."""
        return SUPPORTED_LANGUAGES.get(code, "English")

    @staticmethod
    def get_template(key: str, lang: str) -> str:
        """
        Return a pre-canned response template.
        Falls back to English if the requested language is unavailable.
        """
        bucket = _TEMPLATES.get(key, {})
        return bucket.get(lang, bucket.get("en", ""))
