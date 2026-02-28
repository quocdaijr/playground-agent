import uuid
import structlog
from fastapi import APIRouter, HTTPException

from app.core.langgraph.graph import agent_graph
from app.schemas.chat import ChatRequest, ChatResponse
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


@router.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest) -> ChatResponse:
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

        try:
            result = await agent_graph.ainvoke(state)
        except Exception as exc:
            logger.error("graph_invoke_failed", session_id=sid, error=str(exc))
            raise HTTPException(status_code=500, detail="Agent error") from exc

        new_msgs = state_messages_to_dicts(old_count, result)
        await save_turn(db, session, result["turn_count"], new_msgs)
        await maybe_summarise(db, sid, result["turn_count"], session.messages)

        reply = extract_reply(result)
        logger.info("chat_complete", session_id=sid, turn=result["turn_count"])

    return ChatResponse(reply=reply, session_id=sid, turn_count=result["turn_count"])


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
