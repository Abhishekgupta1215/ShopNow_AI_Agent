"""
agent.py — ShopNow AI Voice Agent
Orchestrates:
  • LangChain AgentExecutor with OpenAI tool-calling
  • ConversationBufferMemory for multi-turn dialogue
  • Language detection & mid-conversation code-switching
  • Real-time sentiment analysis & escalation management
"""

from __future__ import annotations

import os
from typing import Optional

from dotenv import load_dotenv
from langchain.agents import AgentExecutor, create_openai_tools_agent
from langchain.memory import ConversationBufferMemory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_openai import ChatOpenAI

from language import LanguageDetector
from sentiment import EscalationManager, SentimentAnalyzer
from tools import SHOPNOW_TOOLS, TOOL_TO_INTENT

load_dotenv()

# ---------------------------------------------------------------------------
# System-prompt template
# ---------------------------------------------------------------------------

_SYSTEM_PROMPT_TMPL = """\
You are Shopi, a warm, empathetic, and highly capable AI voice support agent \
for ShopNow — a leading Indian e-commerce platform.

Your responsibilities:
1. Order Status       — track and communicate order updates
2. Returns & Refunds  — process return requests; explain refund timelines
3. Payment Issues     — resolve failed payments, duplicate charges, pending authorisations
4. Delivery Complaints— handle delays, wrong/damaged items, non-delivery reports
5. Product Queries    — answer questions on specs, pricing, availability

Behavioural guidelines:
• Always greet the customer by name on the first turn.
• Use the appropriate tool BEFORE answering any question that requires live data.
• Never fabricate order or product details — always fetch from tools.
• Be patient and empathetic, especially with frustrated customers.
• Where relevant, acknowledge seasonal/cultural context for Indian customers.
• For major inconveniences, proactively offer compensation (voucher, priority shipping).
• Provide concise but complete responses.
• If asked in a non-English language, continue in that language.

Customer name : {customer_name}
Language rule : {language_instruction}
"""


# ---------------------------------------------------------------------------
# ShopNow Agent (one per session)
# ---------------------------------------------------------------------------

class ShopNowAgent:
    """
    Stateful agent for a single customer session.

    Each call to ``process_message`` returns a structured dict with:
      response, detected_language, intent, sentiment_score,
      escalation_triggered, escalation_brief, resolved,
      code_switching_detected, turn_count
    """

    def __init__(
        self,
        customer_name: str,
        phone_number: Optional[str] = None,
    ) -> None:
        self.customer_name   = customer_name
        self.phone_number    = phone_number
        self.turn_count      = 0
        self.intents_history: list[str] = []
        self.conv_history:    list[str] = []   # plain-text for escalation brief

        self._sentiment  = SentimentAnalyzer(
            frustration_threshold=float(
                os.getenv("ESCALATION_SENTIMENT_THRESHOLD", "-0.4")
            )
        )
        self._escalation = EscalationManager()
        self._lang       = LanguageDetector()

        # Shared memory object — persists across agent rebuilds
        self._memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True,
        )

        self._executor: Optional[AgentExecutor] = None
        self._active_lang_code = "en"

    # ------------------------------------------------------------------
    # Agent construction
    # ------------------------------------------------------------------

    def _build_executor(self, lang_code: str) -> AgentExecutor:
        """
        Build (or rebuild) the AgentExecutor with a language-aware system prompt.
        The shared ``self._memory`` is reused so conversation history is preserved
        even when the language switches mid-session.
        """
        system_content = _SYSTEM_PROMPT_TMPL.format(
            customer_name=self.customer_name,
            language_instruction=LanguageDetector.get_language_instruction(lang_code),
        )

        prompt = ChatPromptTemplate.from_messages([
            ("system", system_content),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{input}"),
            MessagesPlaceholder(variable_name="agent_scratchpad"),
        ])

        llm = ChatOpenAI(
            model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
            temperature=0.3,
            api_key=os.getenv("OPENAI_API_KEY"),
        )

        agent = create_openai_tools_agent(llm=llm, tools=SHOPNOW_TOOLS, prompt=prompt)

        return AgentExecutor(
            agent=agent,
            tools=SHOPNOW_TOOLS,
            memory=self._memory,
            verbose=bool(os.getenv("AGENT_VERBOSE", "false").lower() == "true"),
            handle_parsing_errors=True,
            max_iterations=6,
        )

    def _ensure_executor(self, lang_code: str) -> None:
        """Lazily construct or refresh the executor when language changes."""
        if self._executor is None or lang_code != self._active_lang_code:
            self._executor = self._build_executor(lang_code)
            self._active_lang_code = lang_code

    # ------------------------------------------------------------------
    # Intent detection (post-hoc from tool names in output)
    # ------------------------------------------------------------------

    def _infer_intent(self, response_text: str, tools_used: list[str]) -> str:
        # Prefer explicit tool name from AgentExecutor intermediate steps
        for tool_name in tools_used:
            if tool_name in TOOL_TO_INTENT:
                return TOOL_TO_INTENT[tool_name]

        # Fallback: keyword scan on the response
        lower = response_text.lower()
        if any(kw in lower for kw in ("order", "track", "shipped", "dispatch")):
            return "order_status"
        if any(kw in lower for kw in ("return", "refund", "exchange", "replace")):
            return "returns_refunds"
        if any(kw in lower for kw in ("payment", "charge", "billing", "debit")):
            return "payment_issues"
        if any(kw in lower for kw in ("deliver", "courier", "damaged", "wrong item")):
            return "delivery_complaints"
        if any(kw in lower for kw in ("product", "specification", "price", "stock")):
            return "product_queries"
        return "general_inquiry"

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def process_message(self, user_message: str) -> dict:
        """
        Process one customer message and return a full result dict.

        Sequence:
          1. Detect language  (and check for code-switching)
          2. Analyse sentiment
          3. Check escalation threshold
          4. Ensure agent executor is built / refreshed
          5. Invoke LangChain agent
          6. If escalation is triggered, prepend notice & generate brief
          7. Infer intent; determine resolution flag
        """
        self.turn_count += 1

        # --- Language ---
        lang_code      = self._lang.detect(user_message)
        code_switching = self._lang.detect_code_switching()

        # --- Sentiment ---
        lang_name_for_lexicon = LanguageDetector.get_language_name(lang_code).lower()
        sentiment_score       = self._sentiment.analyze(user_message, lang_name_for_lexicon)

        # --- Escalation check (snapshot before calling should_escalate) ---
        was_escalated   = self._sentiment.escalation_triggered
        should_escalate = self._sentiment.should_escalate()
        newly_escalated = should_escalate and not was_escalated

        escalation_brief: Optional[dict] = None
        if newly_escalated:
            brief_obj = self._escalation.generate_brief(
                customer_name=self.customer_name,
                intents_detected=self.intents_history,
                conversation_history=self.conv_history,
                sentiment_score=sentiment_score,
                sentiment_analyzer=self._sentiment,
            )
            escalation_brief = brief_obj.to_dict()

        # --- Build / refresh executor ---
        self._ensure_executor(lang_code)

        # --- Run agent ---
        try:
            result       = self._executor.invoke({"input": user_message})
            response     = result.get(
                "output",
                "I apologise — I encountered a technical issue. "
                "Please hold while I transfer you to a specialist.",
            )
            tools_used = [
                step[0].tool
                for step in result.get("intermediate_steps", [])
                if hasattr(step[0], "tool")
            ]
        except Exception as exc:  # noqa: BLE001
            response     = (
                "I'm sorry, I ran into a technical problem processing your request. "
                "A human specialist will be in touch with you shortly."
            )
            tools_used   = []
            newly_escalated = True
            escalation_brief = escalation_brief or {
                "error": str(exc),
                "customer_name": self.customer_name,
            }

        # --- Prepend escalation notice if this turn triggered it ---
        if newly_escalated:
            notice   = LanguageDetector.get_template("escalation_notice", lang_code)
            response = f"{notice}\n\n{response}"

        # --- Track history ---
        self.conv_history.append(f"Customer: {user_message}")
        self.conv_history.append(f"Agent   : {response}")

        # --- Infer intent ---
        intent = self._infer_intent(response, tools_used)
        self.intents_history.append(intent)

        # --- Resolution heuristic ---
        resolved = (not should_escalate) and (sentiment_score > -0.25)

        return {
            "response":               response,
            "detected_language":      LanguageDetector.get_language_name(lang_code),
            "intent":                 intent,
            "sentiment_score":        round(sentiment_score, 3),
            "escalation_triggered":   newly_escalated,
            "escalation_brief":       escalation_brief,
            "resolved":               resolved,
            "code_switching_detected": code_switching,
            "turn_count":             self.turn_count,
        }

    def get_session_summary(self) -> dict:
        """Return a summary snapshot of the entire session."""
        return {
            "customer_name":    self.customer_name,
            "total_turns":      self.turn_count,
            "intents_covered":  list(dict.fromkeys(self.intents_history)),
            "dominant_language": LanguageDetector.get_language_name(
                self._lang.get_dominant_language()
            ),
            "sentiment_trend":  self._sentiment.get_sentiment_trend(),
            "final_sentiment":  round(self._sentiment.get_rolling_sentiment(), 3),
            "escalated":        self._sentiment.escalation_triggered,
        }
