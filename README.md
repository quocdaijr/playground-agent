# PlaygroundAgent

A **reference implementation** of a production-ready conversational AI agent.
Use this as a starting point — fork it, rename it, and extend it for your own use case.

> This is intentionally a generic testing agent. It demonstrates patterns,
> not a finished product.

## Stack

| Layer | Technology |
|-------|-----------|
| Agent | LangGraph `StateGraph` with conditional tool routing |
| API | FastAPI + async lifespan + request-ID middleware |
| LLM | Claude Sonnet via `langchain-anthropic` |
| DB | PostgreSQL + SQLAlchemy 2.0 async + asyncpg |
| Migrations | Alembic (async `env.py`) |
| HTTP client | httpx `AsyncClient` with tenacity retry |
| Logging | structlog (JSON in prod, console in dev) |

## What this demonstrates

- **LangGraph tool routing** — agent decides when to call tools vs respond directly
- **External HTTP tools** — async httpx with retry, shared client, structured errors
- **Conversation memory** — auto-summarise every 10 turns, inject into system prompt
- **PostgreSQL persistence** — sessions, messages, summaries with Alembic migrations
- **Production patterns** — structlog, request IDs, lifespan hooks, graceful shutdown

## Project Structure

```
playground-agent/
├── app/
│   ├── api/v1/chatbot.py     # REST endpoints
│   ├── core/
│   │   ├── config.py         # Pydantic Settings
│   │   ├── langgraph/
│   │   │   ├── graph.py      # StateGraph definition
│   │   │   └── tools.py      # Tool registry (local + external)
│   │   └── prompts/system.md # System prompt template
│   ├── models/session.py     # SQLAlchemy ORM models
│   ├── services/
│   │   ├── database.py       # Async engine + session factory
│   │   ├── session_repo.py   # DB operations
│   │   └── summary.py        # Auto-summarisation service
│   └── main.py               # App factory + lifespan
├── migrations/               # Alembic versions
├── evals/evaluator.py        # Test runner
├── docker-compose.yml
└── pyproject.toml
```

## Tools

| Tool | Type | API | Auth |
|------|------|-----|------|
| `calculator` | local | — | — |
| `get_current_time` | local | — | — |
| `word_counter` | local | — | — |
| `list_operations` | local | — | — |
| `web_search` | external | DuckDuckGo Instant Answer | none |
| `get_weather` | external | Open-Meteo | none |
| `currency_exchange` | external | Frankfurter / ECB | none |
| `ip_lookup` | external | ip-api.com | none |

## Quick Start

```bash
cp .env.example .env
# Set ANTHROPIC_API_KEY in .env

make up
# → http://localhost:8000/docs
```

## API

| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/api/v1/chatbot/chat` | Send a message |
| `GET` | `/api/v1/chatbot/history/{id}` | Message history |
| `GET` | `/api/v1/chatbot/summaries/{id}` | Auto-generated summaries |
| `DELETE` | `/api/v1/chatbot/history/{id}` | Clear session |
| `GET` | `/health` | Health check |

## Customising

To make this your own:
1. **Rename** — find/replace `PlaygroundAgent` / `playground-agent` / `playground_agent`
2. **System prompt** — edit `app/core/prompts/system.md`
3. **Add tools** — add `@tool` functions in `app/core/langgraph/tools.py`, append to `TOOLS`
4. **Schema changes** — edit models, then `make migrate-new`

## LangGraph Flow

```
START → [agent] ──► has tool_calls? ──► [tools] ─┐
                │                                  └──► [agent]
                │ no
                ▼
               END
```
