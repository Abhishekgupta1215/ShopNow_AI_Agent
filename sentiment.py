"""
sentiment.py — ShopNow AI Voice Agent
Real-time keyword-based sentiment analysis and escalation management.
Tracks rolling sentiment per session and generates structured escalation briefs
for human agents when frustration crosses the configured threshold.
"""

from __future__ import annotations

from dataclasses import dataclass


# ---------------------------------------------------------------------------
# Lexicons  (extend per language as needed)
# ---------------------------------------------------------------------------

_FRUSTRATION: dict[str, list[str]] = {
    "english": [
        "angry", "furious", "terrible", "horrible", "awful", "disgusting",
        "frustrated", "useless", "pathetic", "incompetent", "ridiculous",
        "unacceptable", "rubbish", "trash", "worst", "never again",
        "wasted", "scam", "cheated", "fraud", "refund now", "cancel now",
        "this is ridiculous", "waste of money", "poor service", "very bad",
        "already waiting", "still not received", "not acceptable",
    ],
    "hindi": [
        "बेकार", "खराब", "घटिया", "धोखा", "बर्बाद", "नाराज़",
        "गुस्सा", "बकवास", "निकम्मा", "चोर", "झूठ", "ठगी",
    ],
    "tamil": [
        "மோசமான", "குப்பை", "ஏமாற்றம்", "கோபம்", "வீண்",
        "மோசம்", "ஏமாற்று",
    ],
}

_POSITIVE: dict[str, list[str]] = {
    "english": [
        "thanks", "thank you", "great", "excellent", "perfect", "happy",
        "satisfied", "good", "wonderful", "helpful", "awesome", "appreciate",
    ],
    "hindi": ["धन्यवाद", "शुक्रिया", "बढ़िया", "अच्छा", "संतुष्ट", "खुश"],
    "tamil": ["நன்றி", "நல்லது", "மகிழ்ச்சி", "சிறப்பு"],
}


# ---------------------------------------------------------------------------
# Escalation Brief
# ---------------------------------------------------------------------------

@dataclass
class EscalationBrief:
    customer_name: str
    issue_summary: str
    conversation_history: list[str]
    sentiment_score: float
    recommended_tone: str
    priority: str          # "low" | "medium" | "high" | "critical"
    intents_detected: list[str]

    def to_dict(self) -> dict:
        return {
            "customer_name":      self.customer_name,
            "issue_summary":      self.issue_summary,
            "conversation_history": self.conversation_history,
            "sentiment_score":    self.sentiment_score,
            "recommended_tone":   self.recommended_tone,
            "priority":           self.priority,
            "intents_detected":   self.intents_detected,
        }


# ---------------------------------------------------------------------------
# Sentiment Analyser
# ---------------------------------------------------------------------------

class SentimentAnalyzer:
    """
    Lightweight, real-time sentiment analyser for voice-agent sessions.

    Scoring logic:
    • Frustration keywords  → −0.25 each (capped at −0.75)
    • Positive keywords     → +0.20 each (capped at +0.60)
    • Excessive exclamation → up to −0.20
    • High CAPS ratio       → up to −0.20
    • Repeated questions    → up to −0.15

    Final score is clamped to [−1.0, +1.0].
    A rolling average over the last `window_size` messages is used to decide
    whether to escalate the call.
    """

    def __init__(
        self,
        frustration_threshold: float = -0.4,
        window_size: int = 5,
    ) -> None:
        self.frustration_threshold = frustration_threshold
        self.window_size = window_size
        self.sentiment_history: list[float] = []
        self.message_count: int = 0
        self.escalation_triggered: bool = False

    # ------------------------------------------------------------------
    # Core analysis
    # ------------------------------------------------------------------

    def analyze(self, text: str, language: str = "english") -> float:
        """
        Return a sentiment score in [−1.0, +1.0] for the given text.
        Updates internal history so rolling calculations stay fresh.
        """
        text_lower = text.lower()
        score = 0.0

        # Lexicon hits
        frustration_phrases = (
            _FRUSTRATION.get(language, []) + _FRUSTRATION["english"]
        )
        positive_phrases = (
            _POSITIVE.get(language, []) + _POSITIVE["english"]
        )

        frustration_hits = sum(
            1 for phrase in frustration_phrases if phrase.lower() in text_lower
        )
        positive_hits = sum(
            1 for phrase in positive_phrases if phrase.lower() in text_lower
        )

        # Formatting signals
        exclamations  = min(text.count("!") * 0.05, 0.20)
        caps_ratio    = sum(1 for c in text if c.isupper()) / max(len(text), 1)
        caps_penalty  = min(caps_ratio * 0.30, 0.20) if caps_ratio > 0.35 else 0.0
        q_count       = text.count("?")
        q_penalty     = min(q_count * 0.05, 0.15) if q_count > 2 else 0.0

        score -= min(frustration_hits * 0.25, 0.75)
        score += min(positive_hits * 0.20, 0.60)
        score -= exclamations
        score -= caps_penalty
        score -= q_penalty

        score = max(-1.0, min(1.0, score))

        self.sentiment_history.append(score)
        self.message_count += 1
        return score

    # ------------------------------------------------------------------
    # Rolling statistics
    # ------------------------------------------------------------------

    def get_rolling_sentiment(self) -> float:
        """Average sentiment over the last `window_size` messages."""
        if not self.sentiment_history:
            return 0.0
        window = self.sentiment_history[-self.window_size :]
        return sum(window) / len(window)

    def get_sentiment_trend(self) -> str:
        """Returns 'improving', 'stable', or 'deteriorating'."""
        history = self.sentiment_history
        if len(history) < 3:
            return "stable"
        mid = len(history) // 2
        early  = sum(history[:mid]) / mid
        recent = sum(history[mid:]) / len(history[mid:])
        diff   = recent - early
        if diff > 0.15:
            return "improving"
        if diff < -0.15:
            return "deteriorating"
        return "stable"

    # ------------------------------------------------------------------
    # Escalation decision
    # ------------------------------------------------------------------

    def should_escalate(self) -> bool:
        """
        Return True if the conversation should be escalated to a human agent.
        Once triggered the flag stays True for the session lifetime.
        """
        if self.escalation_triggered:
            return True
        if not self.sentiment_history:
            return False

        rolling = self.get_rolling_sentiment()

        # Sustained frustration
        if rolling < self.frustration_threshold:
            self.escalation_triggered = True
            return True

        # Two consecutive very-negative messages
        if (
            len(self.sentiment_history) >= 2
            and all(s < -0.65 for s in self.sentiment_history[-2:])
        ):
            self.escalation_triggered = True
            return True

        return False


# ---------------------------------------------------------------------------
# Escalation Manager
# ---------------------------------------------------------------------------

class EscalationManager:
    """
    Generates structured escalation briefs consumed by human agents.
    Each brief includes: customer name, issue summary, conversation snapshot,
    sentiment score, recommended tone, and priority level.
    """

    _TONE_MAP: dict[str, str] = {
        "critical": (
            "Use a deeply empathetic and apologetic tone. Acknowledge the customer's "
            "frustration explicitly. Offer immediate resolution with concrete timelines. "
            "Consider proactive compensation (voucher, free shipping upgrade)."
        ),
        "high": (
            "Approach with empathy and patience. Prioritise active listening. "
            "Propose clear, actionable next steps. Avoid all defensive language."
        ),
        "medium": (
            "Be professional and understanding. Guide the conversation toward "
            "resolution. Confirm each step clearly before moving on."
        ),
        "low": (
            "Standard professional tone. Focus on quick resolution and a positive close."
        ),
    }

    _INTENT_DESCRIPTIONS: dict[str, str] = {
        "order_status":          "inquired about their order status",
        "returns_refunds":       "requested a return or refund",
        "payment_issues":        "reported a payment problem",
        "delivery_complaints":   "complained about a delivery issue",
        "product_queries":       "asked about a product",
        "general_inquiry":       "made a general inquiry",
    }

    def generate_brief(
        self,
        customer_name: str,
        intents_detected: list[str],
        conversation_history: list[str],
        sentiment_score: float,
        sentiment_analyzer: SentimentAnalyzer,
    ) -> EscalationBrief:
        priority        = self._determine_priority(sentiment_score, sentiment_analyzer)
        issue_summary   = self._build_summary(intents_detected, conversation_history)
        recommended_tone = self._TONE_MAP[priority]

        return EscalationBrief(
            customer_name=customer_name,
            issue_summary=issue_summary,
            conversation_history=conversation_history[-6:],   # latest 6 turns
            sentiment_score=round(sentiment_score, 3),
            recommended_tone=recommended_tone,
            priority=priority,
            intents_detected=list(set(intents_detected)),
        )

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _determine_priority(
        self,
        score: float,
        analyzer: SentimentAnalyzer,
    ) -> str:
        rolling = analyzer.get_rolling_sentiment()
        if score < -0.70 or rolling < -0.60:
            return "critical"
        if score < -0.50:
            return "high"
        if score < -0.30:
            return "medium"
        return "low"

    def _build_summary(
        self,
        intents: list[str],
        history: list[str],
    ) -> str:
        seen = list(dict.fromkeys(intents))   # deduplicate, preserve order
        descriptions = [
            self._INTENT_DESCRIPTIONS.get(i, i) for i in seen
        ]
        if not descriptions:
            snippet = history[-1][:120] if history else "no details available"
            return f"Customer requires assistance. Last message: {snippet}"

        joined = " and ".join(descriptions)
        return f"Customer has {joined}."
