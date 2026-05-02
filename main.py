"""
main.py — ShopNow AI Voice Agent
FastAPI application entry-point.

Endpoints
---------
POST   /chat                        — send a customer message
GET    /session/{id}/summary        — conversation summary for a session
DELETE /session/{id}                — close and clean up a session
GET    /analytics/daily-report      — daily analytics report
GET    /analytics/available-dates   — dates that have logged data
GET    /health                      — service health-check
"""

from __future__ import annotations

import os
import uuid
from datetime import date
from typing import Optional

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from agent import ShopNowAgent
from analytics import analytics_logger

load_dotenv()

# ---------------------------------------------------------------------------
# App setup
# ---------------------------------------------------------------------------

app = FastAPI(
    title="ShopNow AI Voice Agent",
    description=(
        "AI-powered, multilingual customer-support voice agent for ShopNow. "
        "Handles order status, returns, payments, delivery complaints, and product queries."
    ),
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# In-memory session store  {session_id: ShopNowAgent}
_sessions: dict[str, ShopNowAgent] = {}


# ---------------------------------------------------------------------------
# Pydantic schemas
# ---------------------------------------------------------------------------

class ChatRequest(BaseModel):
    session_id: Optional[str] = None
    customer_name: str
    message: str
    phone_number: Optional[str] = None


class ChatResponse(BaseModel):
    session_id: str
    response: str
    detected_language: str
    intent: str
    sentiment_score: float
    escalation_triggered: bool
    escalation_brief: Optional[dict] = None
    code_switching_detected: bool = False
    turn_count: int = 0


class SessionSummaryResponse(BaseModel):
    session_id: str
    summary: dict


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.post("/chat", response_model=ChatResponse, tags=["Voice Agent"])
async def chat(request: ChatRequest) -> ChatResponse:
    """
    Send a customer message to the ShopNow AI agent.

    - If **session_id** is omitted (or unknown), a new session is created and
      the ID is returned so the client can maintain conversation continuity.
    - The same endpoint handles all intents; the agent picks the right tool.
    """
    # Resolve or create session
    sid = request.session_id
    if not sid or sid not in _sessions:
        sid = sid or str(uuid.uuid4())
        _sessions[sid] = ShopNowAgent(
            customer_name=request.customer_name,
            phone_number=request.phone_number,
        )
        analytics_logger.start_session(sid)

    agent  = _sessions[sid]
    result = agent.process_message(request.message)

    # Persist to analytics
    analytics_logger.log_interaction(
        session_id=sid,
        intent=result["intent"],
        sentiment_score=result["sentiment_score"],
        resolved=result.get("resolved", False),
        language=result["detected_language"],
        escalated=result.get("escalation_triggered", False),
    )

    return ChatResponse(
        session_id=sid,
        response=result["response"],
        detected_language=result["detected_language"],
        intent=result["intent"],
        sentiment_score=result["sentiment_score"],
        escalation_triggered=result["escalation_triggered"],
        escalation_brief=result.get("escalation_brief"),
        code_switching_detected=result.get("code_switching_detected", False),
        turn_count=result.get("turn_count", 0),
    )


@app.get(
    "/session/{session_id}/summary",
    response_model=SessionSummaryResponse,
    tags=["Sessions"],
)
async def get_session_summary(session_id: str) -> SessionSummaryResponse:
    """Return a high-level summary of an active conversation session."""
    if session_id not in _sessions:
        raise HTTPException(status_code=404, detail="Session not found.")
    return SessionSummaryResponse(
        session_id=session_id,
        summary=_sessions[session_id].get_session_summary(),
    )


@app.delete("/session/{session_id}", tags=["Sessions"])
async def end_session(session_id: str) -> dict:
    """
    Gracefully end a conversation session and free its memory.
    Analytics data for the session is retained on disk.
    """
    if session_id not in _sessions:
        raise HTTPException(status_code=404, detail="Session not found.")
    analytics_logger.close_session(session_id)
    del _sessions[session_id]
    return {"ok": True, "message": "Session ended.", "session_id": session_id}


@app.get("/analytics/daily-report", tags=["Analytics"])
async def daily_report(report_date: Optional[str] = None) -> dict:
    """
    Generate a daily analytics report.

    - **report_date** (optional, ISO format YYYY-MM-DD) — defaults to today.

    The report includes call volume by intent, per-intent resolution rates,
    sentiment trends, language distribution, escalation rate, and hourly
    heatmap.
    """
    target = report_date or str(date.today())
    return analytics_logger.generate_daily_report(target)


@app.get("/analytics/available-dates", tags=["Analytics"])
async def available_dates() -> dict:
    """List all dates for which interaction data has been logged."""
    return {"dates": analytics_logger.get_all_report_dates()}


@app.get("/health", tags=["System"])
async def health_check() -> dict:
    """Service liveness probe."""
    return {
        "status":          "healthy",
        "service":         "ShopNow AI Voice Agent",
        "version":         "1.0.0",
        "active_sessions": len(_sessions),
    }


# ---------------------------------------------------------------------------
# Dev runner
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "main:app",
        host=os.getenv("APP_HOST", "0.0.0.0"),
        port=int(os.getenv("APP_PORT", "8000")),
        reload=True,
        log_level=os.getenv("LOG_LEVEL", "info").lower(),
    )
