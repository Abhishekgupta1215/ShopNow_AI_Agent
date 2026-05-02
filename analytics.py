"""
analytics.py — ShopNow AI Voice Agent
Thread-safe interaction logger and daily report generator.

Every call interaction is persisted to a JSON log file.
The daily report covers:
  • Total call volume and resolution rate
  • Call volume broken down by intent
  • Per-intent resolution rates and average sentiment
  • Language distribution
  • Hourly call-volume heat-map
  • Sentiment trend (hourly averages)
  • Escalation rate
"""

from __future__ import annotations

import json
import os
import threading
from collections import defaultdict
from dataclasses import asdict, dataclass
from datetime import date, datetime
from typing import Optional


# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------

@dataclass
class InteractionLog:
    session_id: str
    timestamp: str
    date: str
    intent: str
    sentiment_score: float
    resolved: bool
    language: str
    escalated: bool = False


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _sentiment_label(score: float) -> str:
    if score >= 0.30:
        return "Very Positive"
    if score >= 0.10:
        return "Positive"
    if score >= -0.10:
        return "Neutral"
    if score >= -0.30:
        return "Slightly Negative"
    if score >= -0.50:
        return "Negative"
    return "Very Negative"


def _safe_avg(values: list[float]) -> float:
    return round(sum(values) / len(values), 3) if values else 0.0


# ---------------------------------------------------------------------------
# Analytics Logger
# ---------------------------------------------------------------------------

class AnalyticsLogger:
    """
    Persists every interaction to ``ANALYTICS_LOG_FILE`` (JSON-Lines style
    wrapped in a JSON array) and provides a daily aggregation report.

    Thread-safe: a ``threading.Lock`` guards all write operations.
    """

    def __init__(self, log_file: Optional[str] = None) -> None:
        self.log_file = log_file or os.getenv(
            "ANALYTICS_LOG_FILE", "analytics_logs.json"
        )
        self._lock = threading.Lock()
        self._logs: list[InteractionLog] = []
        self._session_start_times: dict[str, datetime] = {}
        self._load_existing()

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def _load_existing(self) -> None:
        if not os.path.exists(self.log_file):
            return
        try:
            with open(self.log_file, "r", encoding="utf-8") as fh:
                raw = json.load(fh)
            self._logs = [InteractionLog(**entry) for entry in raw]
        except (json.JSONDecodeError, TypeError, KeyError):
            self._logs = []

    def _persist(self) -> None:
        """Write current log list to disk (caller must hold _lock)."""
        with open(self.log_file, "w", encoding="utf-8") as fh:
            json.dump([asdict(log) for log in self._logs], fh, indent=2)

    # ------------------------------------------------------------------
    # Session lifecycle
    # ------------------------------------------------------------------

    def start_session(self, session_id: str) -> None:
        self._session_start_times[session_id] = datetime.now()

    def close_session(self, session_id: str) -> None:
        self._session_start_times.pop(session_id, None)

    # ------------------------------------------------------------------
    # Logging
    # ------------------------------------------------------------------

    def log_interaction(
        self,
        *,
        session_id: str,
        intent: str,
        sentiment_score: float,
        resolved: bool,
        language: str,
        escalated: bool = False,
    ) -> None:
        """Append a single interaction record and flush to disk."""
        now = datetime.now()
        entry = InteractionLog(
            session_id=session_id,
            timestamp=now.isoformat(),
            date=str(now.date()),
            intent=intent,
            sentiment_score=round(sentiment_score, 3),
            resolved=resolved,
            language=language,
            escalated=escalated,
        )
        with self._lock:
            self._logs.append(entry)
            self._persist()

    # ------------------------------------------------------------------
    # Report generation
    # ------------------------------------------------------------------

    def generate_daily_report(self, target_date: Optional[str] = None) -> dict:
        """
        Produce a comprehensive daily analytics report for *target_date*
        (ISO format: YYYY-MM-DD).  Defaults to today.

        Report structure
        ----------------
        date, generated_at
        summary:
            total_interactions, total_resolved, overall_resolution_rate,
            escalations, escalation_rate, avg_sentiment, sentiment_status
        intent_breakdown:
            per intent → total_calls, resolved, unresolved,
                         resolution_rate, avg_sentiment, sentiment_label
        language_distribution:
            per language → count, percentage
        sentiment_trend:
            hourly average sentiment  { "HH:00": score }
        hourly_call_volume:
            { "HH": count }
        peak_hour
        """
        target_date = target_date or str(date.today())
        daily = [log for log in self._logs if log.date == target_date]

        if not daily:
            return {
                "date": target_date,
                "generated_at": datetime.now().isoformat(),
                "summary": "No interactions recorded for this date.",
                "total_interactions": 0,
            }

        total = len(daily)
        total_resolved   = sum(1 for l in daily if l.resolved)
        total_escalated  = sum(1 for l in daily if l.escalated)
        all_sentiments   = [l.sentiment_score for l in daily]
        avg_sentiment    = _safe_avg(all_sentiments)

        # --- Intent breakdown ---
        intent_counts:     defaultdict[str, int]         = defaultdict(int)
        intent_resolved:   defaultdict[str, int]         = defaultdict(int)
        intent_sentiments: defaultdict[str, list[float]] = defaultdict(list)

        for log in daily:
            intent_counts[log.intent]           += 1
            intent_sentiments[log.intent].append(log.sentiment_score)
            if log.resolved:
                intent_resolved[log.intent] += 1

        intent_breakdown: dict[str, dict] = {}
        for intent, count in intent_counts.items():
            resolved  = intent_resolved[intent]
            sentiments = intent_sentiments[intent]
            avg_s     = _safe_avg(sentiments)
            intent_breakdown[intent] = {
                "total_calls":      count,
                "resolved":         resolved,
                "unresolved":       count - resolved,
                "resolution_rate":  f"{resolved / count * 100:.1f}%",
                "avg_sentiment":    avg_s,
                "sentiment_label":  _sentiment_label(avg_s),
            }

        # --- Language distribution ---
        lang_counts: defaultdict[str, int] = defaultdict(int)
        for log in daily:
            lang_counts[log.language] += 1

        language_distribution = {
            lang: {
                "count":      cnt,
                "percentage": f"{cnt / total * 100:.1f}%",
            }
            for lang, cnt in sorted(
                lang_counts.items(), key=lambda x: x[1], reverse=True
            )
        }

        # --- Hourly breakdowns ---
        hourly_volume:    defaultdict[int, int]         = defaultdict(int)
        hourly_sentiment: defaultdict[int, list[float]] = defaultdict(list)

        for log in daily:
            hour = datetime.fromisoformat(log.timestamp).hour
            hourly_volume[hour]             += 1
            hourly_sentiment[hour].append(log.sentiment_score)

        hourly_call_volume = {
            f"{h:02d}": cnt
            for h, cnt in sorted(hourly_volume.items())
        }
        sentiment_trend = {
            f"{h:02d}:00": _safe_avg(scores)
            for h, scores in sorted(hourly_sentiment.items())
        }
        peak_hour = (
            f"{max(hourly_volume, key=hourly_volume.get):02d}:00"
            if hourly_volume else "N/A"
        )

        return {
            "date":         target_date,
            "generated_at": datetime.now().isoformat(),
            "summary": {
                "total_interactions":    total,
                "total_resolved":        total_resolved,
                "overall_resolution_rate": f"{total_resolved / total * 100:.1f}%",
                "escalations":           total_escalated,
                "escalation_rate":       f"{total_escalated / total * 100:.1f}%",
                "avg_sentiment_score":   avg_sentiment,
                "sentiment_status":      _sentiment_label(avg_sentiment),
            },
            "intent_breakdown":       intent_breakdown,
            "language_distribution":  language_distribution,
            "sentiment_trend":        sentiment_trend,
            "hourly_call_volume":     hourly_call_volume,
            "peak_hour":              peak_hour,
        }

    def get_all_report_dates(self) -> list[str]:
        """Return all unique dates that have logged data, newest first."""
        return sorted({log.date for log in self._logs}, reverse=True)

    # kept for backwards-compat (public alias)
    def get_all_reports(self) -> list[str]:
        return self.get_all_report_dates()


# ---------------------------------------------------------------------------
# Module-level singleton (imported by main.py and agent.py)
# ---------------------------------------------------------------------------

analytics_logger = AnalyticsLogger()
