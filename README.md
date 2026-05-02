# ShopNow AI Voice Agent

An AI-powered, multilingual customer-support voice agent for the **ShopNow** e-commerce brand, built with **LangChain** and **FastAPI**.

---

## Features

| Feature | Implementation |
|---|---|
| **5 support intents** | Order Status · Returns/Refunds · Payment Issues · Delivery Complaints · Product Queries |
| **Multi-turn dialogue** | `ConversationBufferMemory` preserves context across every turn |
| **Multilingual** | English · Hindi · Tamil — Unicode script detection + romanised-keyword fallback, mid-conversation code-switching handled automatically |
| **Sentiment & Escalation** | Rolling keyword-based scorer; when score falls below threshold, generates a structured **Escalation Brief** (customer name, issue summary, recommended tone, priority) |
| **Mock database** | `database.py` — 6 orders, 7 products, customer records, return-eligibility logic |
| **Analytics** | Per-interaction logging → daily report: call volume by intent, resolution rates, sentiment trends, language mix, hourly heatmap |
| **Docker** | Single-command packaging via `Dockerfile` |

---

## Project Structure

```
ShopNow_AI_Agent/
├── main.py          # FastAPI application & API routes
├── agent.py         # LangChain agent (memory + tools + escalation)
├── tools.py         # 5 LangChain @tool functions (one per intent)
├── database.py      # Mock backend — orders, products, returns
├── sentiment.py     # SentimentAnalyzer + EscalationManager
├── language.py      # LanguageDetector (en / hi / ta)
├── analytics.py     # AnalyticsLogger + daily report generator
├── .env             # API keys & configuration (fill in before running)
├── requirements.txt # Python dependencies
├── Dockerfile       # Container packaging
└── .gitignore
```

---

## Quick Start

### 1 — Clone & configure

```bash
git clone https://github.com/Abhishekgupta1215/ShopNow_AI_Agent.git
cd ShopNow_AI_Agent
```

Edit `.env` and set your OpenAI API key:

```ini
OPENAI_API_KEY=sk-...
OPENAI_MODEL=gpt-4o-mini
```

### 2 — Install dependencies

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS / Linux
source .venv/bin/activate

pip install -r requirements.txt
```

### 3 — Run

```bash
uvicorn main:app --reload
# API docs → http://localhost:8000/docs
```

---

## Docker

```bash
# Build
docker build -t shopnow-agent .

# Run (secrets passed at runtime, never baked in)
docker run -p 8000:8000 --env-file .env shopnow-agent
```

---

## API Reference

### `POST /chat`
Send a customer message.

```json
{
  "customer_name": "Rahul Sharma",
  "message": "Where is my order ORD001?",
  "session_id": null,
  "phone_number": "+91-9876543210"
}
```

**Response:**

```json
{
  "session_id": "uuid-...",
  "response": "Order ORD001 Status: SHIPPED ...",
  "detected_language": "English",
  "intent": "order_status",
  "sentiment_score": 0.0,
  "escalation_triggered": false,
  "escalation_brief": null,
  "code_switching_detected": false,
  "turn_count": 1
}
```

### `GET /analytics/daily-report?report_date=2026-03-17`
Returns call volume by intent, resolution rates, sentiment trend, language distribution.

### `GET /session/{id}/summary`
High-level session stats (intents, language, sentiment trend, escalation status).

### `DELETE /session/{id}`
End and free a session.

### `GET /health`
Liveness probe.

---

## Escalation Brief Example

When customer frustration crosses the threshold, the response includes:

```json
{
  "customer_name": "Rahul Sharma",
  "issue_summary": "Customer has complained about a delivery issue.",
  "conversation_history": ["Customer: ...", "Agent: ..."],
  "sentiment_score": -0.75,
  "recommended_tone": "Use a deeply empathetic and apologetic tone ...",
  "priority": "critical",
  "intents_detected": ["delivery_complaints"]
}
```

---

## Configuration (`.env`)

| Variable | Default | Description |
|---|---|---|
| `OPENAI_API_KEY` | — | **Required** — your OpenAI key |
| `OPENAI_MODEL` | `gpt-4o-mini` | Any OpenAI chat-completion model |
| `ESCALATION_SENTIMENT_THRESHOLD` | `-0.4` | Rolling score below which to escalate |
| `ANALYTICS_LOG_FILE` | `analytics_logs.json` | Path to the JSON analytics store |
| `APP_HOST` | `0.0.0.0` | Uvicorn bind host |
| `APP_PORT` | `8000` | Uvicorn port |
| `LOG_LEVEL` | `info` | Uvicorn log level |
| `AGENT_VERBOSE` | `false` | Set `true` to log LangChain tool steps |

---

## Tech Stack

- **[FastAPI](https://fastapi.tiangolo.com/)** — async REST API
- **[LangChain](https://python.langchain.com/)** — agent orchestration & memory
- **[OpenAI](https://platform.openai.com/)** — GPT-4o-mini as the reasoning backbone
- **Python 3.11**
