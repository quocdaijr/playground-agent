import uuid

import structlog
from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse

from app.core.langgraph.graph import agent_graph, get_graph
from app.core.langgraph.research_graph import research_graph
from app.core.sse import stream_graph_events
from app.schemas.chat import (
    ChatRequest,
    ChatResponse,
    ResearchRequest,
    ResearchResponse,
)
from app.services.database import get_session
from app.services.session_repo import (
    delete_session,
    get_or_create_session,
    get_session_with_messages,
    save_turn,
)
from app.services.summary import get_all_summaries, get_latest_summary, maybe_summarise
from app.utils.graph import (
    append_user_message,
    build_initial_state,
    db_messages_to_state,
    extract_reply,
    state_messages_to_dicts,
)

router = APIRouter(prefix="/chatbot", tags=["chatbot"])
logger = structlog.get_logger(__name__)


# ── Chat (blocking) ────────────────────────────────────────────────────────────

@router.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest) -> ChatResponse:
    """Standard chat endpoint — awaits full agent response before returning."""
    sid = req.session_id or str(uuid.uuid4())

    async with get_session() as db:
        session = await get_or_create_session(db, sid)
        summary_ctx = await get_latest_summary(db, sid) or ""

        if session.messages:
            old_count = len(session.messages)
            state = db_messages_to_state(session.messages, session.turn_count, summary_ctx)
            state = append_user_message(state, req.message, summary_ctx)
        else:
            old_count = 0
            state = build_initial_state(req.message, summary_ctx)

        graph = get_graph(req.provider)
        try:
            result = await graph.ainvoke(state)
        except Exception as exc:
            logger.error("graph_invoke_failed", session_id=sid, error=str(exc))
            raise HTTPException(status_code=500, detail="Agent error") from exc

        new_msgs = state_messages_to_dicts(old_count, result)
        await save_turn(db, session, result["turn_count"], new_msgs)
        await maybe_summarise(db, sid, result["turn_count"], session.messages)

        reply = extract_reply(result)
        logger.info("chat_complete", session_id=sid, turn=result["turn_count"], provider=req.provider)

    return ChatResponse(reply=reply, session_id=sid, turn_count=result["turn_count"])


# ── Chat (SSE streaming) ───────────────────────────────────────────────────────

@router.post("/chat/stream")
async def chat_stream(req: ChatRequest) -> StreamingResponse:
    """
    SSE streaming endpoint for the chat graph.

    Events emitted: stage, token, tool_call, tool_result, done, error
    Content-Type: text/event-stream

    Client example (JavaScript):
        const res = await fetch('/api/v1/chatbot/chat/stream', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ message: 'Hello', provider: 'anthropic' }),
        });
        const reader = res.body.getReader();
        // read SSE frames from reader...
    """
    sid = req.session_id or str(uuid.uuid4())

    async with get_session() as db:
        session = await get_or_create_session(db, sid)
        summary_ctx = await get_latest_summary(db, sid) or ""

        if session.messages:
            old_count = len(session.messages)
            state = db_messages_to_state(session.messages, session.turn_count, summary_ctx)
            state = append_user_message(state, req.message, summary_ctx)
        else:
            old_count = 0
            state = build_initial_state(req.message, summary_ctx)

    graph = get_graph(req.provider)

    async def persist(final_state: dict) -> None:
        """Persist the completed turn to DB after streaming finishes."""
        async with get_session() as db:
            session = await get_or_create_session(db, sid)
            new_msgs = state_messages_to_dicts(old_count, final_state)
            turn = final_state.get("turn_count", 0)
            await save_turn(db, session, turn, new_msgs)
            await maybe_summarise(db, sid, turn, session.messages)
        logger.info("chat_stream_persisted", session_id=sid, turn=final_state.get("turn_count"))

    async def event_generator():
        async for chunk in stream_graph_events(graph, state, sid, on_complete=persist):
            yield chunk

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
            "Connection": "keep-alive",
        },
    )


# ── Research pipeline (blocking) ───────────────────────────────────────────────

@router.post("/research", response_model=ResearchResponse)
async def research(req: ResearchRequest) -> ResearchResponse:
    """
    5-stage research pipeline — blocking endpoint that returns the full report.

    Pipeline stages: plan → research → synthesize → review → finalize
    """
    sid = req.session_id or str(uuid.uuid4())

    initial_state = {
        "query": req.query,
        "sub_questions": [],
        "search_results": {},
        "draft_report": "",
        "critique": "",
        "final_answer": "",
        "stage": "init",
        "messages": [],
        "provider": req.provider,
    }

    try:
        result = await research_graph.ainvoke(initial_state)
    except Exception as exc:
        logger.error("research_graph_failed", session_id=sid, error=str(exc))
        raise HTTPException(status_code=500, detail="Research pipeline error") from exc

    # Persist a summary of the research run (query + final answer)
    async with get_session() as db:
        session = await get_or_create_session(db, sid)
        turn = len(result.get("sub_questions", [])) + 2  # rough turn count
        msgs = [
            {"role": "human", "content": req.query},
            {"role": "ai", "content": result["final_answer"]},
        ]
        await save_turn(db, session, turn, msgs)

    logger.info("research_complete", session_id=sid, provider=req.provider,
                stages=5, sub_questions=len(result["sub_questions"]))

    return ResearchResponse(
        final_answer=result["final_answer"],
        sub_questions=result["sub_questions"],
        session_id=sid,
        stages_completed=5,
    )


# ── Research pipeline (SSE streaming) ─────────────────────────────────────────

@router.post("/research/stream")
async def research_stream(req: ResearchRequest) -> StreamingResponse:
    """
    SSE streaming endpoint for the research pipeline.

    Emits one ``stage`` event per pipeline stage (plan, research, synthesize,
    review, finalize), plus ``token`` events during each LLM call, and a
    final ``done`` event with the complete answer.
    """
    sid = req.session_id or str(uuid.uuid4())

    initial_state = {
        "query": req.query,
        "sub_questions": [],
        "search_results": {},
        "draft_report": "",
        "critique": "",
        "final_answer": "",
        "stage": "init",
        "messages": [],
        "provider": req.provider,
    }

    async def persist(final_state: dict) -> None:
        async with get_session() as db:
            session = await get_or_create_session(db, sid)
            msgs = [
                {"role": "human", "content": req.query},
                {"role": "ai", "content": final_state.get("final_answer", "")},
            ]
            await save_turn(db, session, 5, msgs)
        logger.info("research_stream_persisted", session_id=sid)

    async def event_generator():
        async for chunk in stream_graph_events(
            research_graph, initial_state, sid, on_complete=persist
        ):
            yield chunk

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
            "Connection": "keep-alive",
        },
    )


# ── History / Summaries / Delete ───────────────────────────────────────────────

@router.get("/history/{session_id}")
async def get_history(session_id: str) -> dict:
    async with get_session() as db:
        session = await get_session_with_messages(db, session_id)
        if not session:
            raise HTTPException(status_code=404, detail="Session not found")
        return {
            "session_id": session_id,
            "turn_count": session.turn_count,
            "messages": [{"role": m.role, "content": m.content} for m in session.messages],
        }


@router.get("/summaries/{session_id}")
async def get_summaries(session_id: str) -> dict:
    """Return all auto-generated summaries for a session."""
    async with get_session() as db:
        session = await get_session_with_messages(db, session_id)
        if not session:
            raise HTTPException(status_code=404, detail="Session not found")
        summaries = await get_all_summaries(db, session_id)
        return {
            "session_id": session_id,
            "summaries": [
                {
                    "id": s.id,
                    "content": s.content,
                    "turns": f"{s.turn_start}-{s.turn_end}",
                    "created_at": s.created_at.isoformat(),
                }
                for s in summaries
            ],
        }


@router.delete("/history/{session_id}")
async def clear_history(session_id: str) -> dict:
    async with get_session() as db:
        deleted = await delete_session(db, session_id)
        if not deleted:
            raise HTTPException(status_code=404, detail="Session not found")
    return {"deleted": session_id}
