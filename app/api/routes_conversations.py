from fastapi import APIRouter, Depends, Request, Form, HTTPException, status
from fastapi.responses import RedirectResponse
from fastapi.templating import Jinja2Templates
from sqlalchemy.orm import Session
from datetime import datetime
import json
import logging

from app.models.database import get_db
from app.core.auth import get_current_user, validate_workspace
from app.models.conversation import Conversation
from app.models.query import QueryLog
from app.models.feedback import QueryFeedback
from app.models.knowledge_base import KnowledgeBase
from app.models.chunk import Chunk
from app.models.document import Document

logger = logging.getLogger(__name__)

router = APIRouter()
templates = Jinja2Templates(directory="app/templates")

ALLOWED_RATINGS = {"helpful", "not_helpful"}
ALLOWED_REASONS = {
    "incorrect_answer",
    "insufficient_evidence",
    "irrelevant_response",
    "citation_problem",
    "other"
}

def sanitize_title(title: str, max_length: int = 120) -> str:
    cleaned = (title or "").strip()
    if not cleaned:
        cleaned = "New Conversation"
    if len(cleaned) > max_length:
        cleaned = cleaned[:max_length].rstrip() + "..."
    return cleaned


@router.get("/conversations")
async def list_conversations(
    request: Request,
    workspace_id: str = None,
    db: Session = Depends(get_db),
    user=Depends(get_current_user)
):
    ws_id = validate_workspace(workspace_id, user)
    
    conversations = (
        db.query(Conversation)
        .filter(Conversation.workspace_id == ws_id)
        .order_by(Conversation.updated_at.desc())
        .all()
    )

    conv_data = []
    for conv in conversations:
        # Get query count and latest query mode
        queries = db.query(QueryLog).filter(QueryLog.conversation_id == conv.id).order_by(QueryLog.timestamp.desc()).all()
        latest_mode = queries[0].mode if queries else "standard"
        kb_name = conv.knowledge_base.name if conv.knowledge_base else "All Knowledge Bases"

        conv_data.append({
            "id": conv.id,
            "title": conv.title,
            "created_at": conv.created_at,
            "updated_at": conv.updated_at,
            "kb_id": conv.kb_id,
            "kb_name": kb_name,
            "query_count": len(queries),
            "latest_mode": latest_mode,
            "owner_id": conv.user_id,
            "is_owner": (conv.user_id == user.id)
        })

    if "application/json" in request.headers.get("accept", ""):
        return {"conversations": conv_data}

    # Fetch knowledge bases for "Start New Conversation" modal/form
    kbs = db.query(KnowledgeBase).filter(KnowledgeBase.workspace_id == ws_id).order_by(KnowledgeBase.created_at.asc()).all()

    return templates.TemplateResponse("conversations.html", {
        "request": request,
        "conversations": conv_data,
        "knowledge_bases": kbs
    })


@router.post("/conversations")
async def create_conversation(
    request: Request,
    title: str = Form(None),
    kb_id: str = Form(None),
    workspace_id: str = Form(None),
    db: Session = Depends(get_db),
    user=Depends(get_current_user)
):
    ws_id = validate_workspace(workspace_id, user)
    
    # If kb_id is provided, verify it belongs to user's workspace
    clean_kb_id = kb_id.strip() if kb_id and kb_id.strip() else None
    if clean_kb_id:
        kb = db.query(KnowledgeBase).filter(KnowledgeBase.id == clean_kb_id, KnowledgeBase.workspace_id == ws_id).first()
        if not kb:
            raise HTTPException(status_code=404, detail="Knowledge base not found or unauthorized")

    final_title = sanitize_title(title)
    conv = Conversation(
        title=final_title,
        workspace_id=ws_id,
        user_id=user.id,
        kb_id=clean_kb_id,
        created_at=datetime.utcnow(),
        updated_at=datetime.utcnow()
    )
    db.add(conv)
    db.commit()
    db.refresh(conv)

    if "application/json" in request.headers.get("accept", ""):
        return {
            "id": conv.id,
            "title": conv.title,
            "workspace_id": conv.workspace_id,
            "kb_id": conv.kb_id,
            "created_at": conv.created_at.isoformat(),
            "updated_at": conv.updated_at.isoformat()
        }

    return RedirectResponse(url=f"/conversations/{conv.id}", status_code=status.HTTP_303_SEE_OTHER)


@router.get("/conversations/{conversation_id}")
async def get_conversation(
    conversation_id: str,
    request: Request,
    workspace_id: str = None,
    db: Session = Depends(get_db),
    user=Depends(get_current_user)
):
    ws_id = validate_workspace(workspace_id, user)
    conv = db.query(Conversation).filter(
        Conversation.id == conversation_id,
        Conversation.workspace_id == ws_id
    ).first()

    if not conv:
        raise HTTPException(status_code=404, detail="Conversation not found or unauthorized")

    # Fetch all queries in chronological order
    queries = (
        db.query(QueryLog)
        .filter(QueryLog.conversation_id == conv.id, QueryLog.workspace_id == ws_id)
        .order_by(QueryLog.timestamp.asc())
        .all()
    )

    query_details = []
    for q in queries:
        # Parse and verify citations against active chunks & documents in workspace
        raw_citations = []
        if q.citations_data:
            try:
                raw_citations = json.loads(q.citations_data)
            except Exception:
                raw_citations = []

        verified_citations = []
        for cit in raw_citations:
            chunk_id = cit.get("chunk_id")
            # Check if chunk and doc still exist in workspace
            chunk_exists = False
            if chunk_id:
                chunk_rec = db.query(Chunk).filter(Chunk.id == chunk_id, Chunk.workspace_id == ws_id).first()
                if chunk_rec:
                    doc_rec = db.query(Document).filter(Document.id == chunk_rec.document_id, Document.workspace_id == ws_id).first()
                    if doc_rec and doc_rec.status != "DELETED":
                        chunk_exists = True

            verified_citations.append({
                "citation_id": cit.get("citation_id", 1),
                "chunk_id": chunk_id,
                "filename": cit.get("filename", "Unknown Document"),
                "doc_type": cit.get("doc_type", ""),
                "page_number": cit.get("page_number"),
                "score": cit.get("score", 0.0),
                "dense_score": cit.get("dense_score", 0.0),
                "bm25_score": cit.get("bm25_score", 0.0),
                "excerpt": cit.get("excerpt", ""),
                "source_available": chunk_exists
            })

        # Fetch feedback for current user
        fb_rec = db.query(QueryFeedback).filter(
            QueryFeedback.query_id == q.id,
            QueryFeedback.user_id == user.id
        ).first()

        feedback_data = None
        if fb_rec:
            feedback_data = {
                "id": fb_rec.id,
                "rating": fb_rec.rating,
                "reason": fb_rec.reason,
                "comment": fb_rec.comment,
                "updated_at": fb_rec.updated_at.isoformat()
            }

        query_details.append({
            "id": q.id,
            "query": q.query,
            "answer": q.answer,
            "mode": q.mode,
            "answer_status": q.answer_status or "SUCCESS",
            "adaptive_outcome": q.adaptive_outcome,
            "latency": q.latency,
            "timestamp": q.timestamp,
            "citations": verified_citations,
            "evaluation": {
                "overall_score": q.evaluation.overall_score,
                "faithfulness_score": q.evaluation.faithfulness_score,
                "context_score": q.evaluation.context_score,
                "relevance_score": q.evaluation.relevance_score,
                "explanation": q.evaluation.explanation
            } if (q.mode == "adaptive" and q.evaluation) else None,
            "feedback": feedback_data
        })

    if "application/json" in request.headers.get("accept", ""):
        return {
            "conversation": {
                "id": conv.id,
                "title": conv.title,
                "kb_id": conv.kb_id,
                "kb_name": conv.knowledge_base.name if conv.knowledge_base else None,
                "created_at": conv.created_at.isoformat(),
                "updated_at": conv.updated_at.isoformat(),
                "is_owner": (conv.user_id == user.id)
            },
            "queries": query_details
        }

    kbs = db.query(KnowledgeBase).filter(KnowledgeBase.workspace_id == ws_id).order_by(KnowledgeBase.created_at.asc()).all()

    return templates.TemplateResponse("conversation_detail.html", {
        "request": request,
        "conversation": conv,
        "queries": query_details,
        "knowledge_bases": kbs
    })


@router.post("/conversations/{conversation_id}/rename")
async def rename_conversation(
    conversation_id: str,
    request: Request,
    title: str = Form(...),
    workspace_id: str = Form(None),
    db: Session = Depends(get_db),
    user=Depends(get_current_user)
):
    ws_id = validate_workspace(workspace_id, user)
    conv = db.query(Conversation).filter(
        Conversation.id == conversation_id,
        Conversation.workspace_id == ws_id
    ).first()

    if not conv:
        raise HTTPException(status_code=404, detail="Conversation not found or unauthorized")

    # Verify user ownership of the conversation
    if conv.user_id != user.id:
        raise HTTPException(status_code=403, detail="Only the conversation owner can rename this conversation")

    cleaned_title = sanitize_title(title)
    if not cleaned_title or cleaned_title == "New Conversation" and not title.strip():
        raise HTTPException(status_code=400, detail="Conversation title cannot be empty")

    conv.title = cleaned_title
    conv.updated_at = datetime.utcnow()
    db.commit()
    db.refresh(conv)

    if "application/json" in request.headers.get("accept", ""):
        return {"status": "ok", "id": conv.id, "title": conv.title}

    return RedirectResponse(url=f"/conversations/{conv.id}", status_code=status.HTTP_303_SEE_OTHER)


@router.post("/conversations/{conversation_id}/delete")
async def delete_conversation(
    conversation_id: str,
    request: Request,
    workspace_id: str = Form(None),
    db: Session = Depends(get_db),
    user=Depends(get_current_user)
):
    ws_id = validate_workspace(workspace_id, user)
    conv = db.query(Conversation).filter(
        Conversation.id == conversation_id,
        Conversation.workspace_id == ws_id
    ).first()

    if not conv:
        raise HTTPException(status_code=404, detail="Conversation not found or unauthorized")

    # Verify user ownership
    if conv.user_id != user.id:
        raise HTTPException(status_code=403, detail="Only the conversation owner can delete this conversation")

    # Cascade delete deletes associated queries and feedback via SQLAlchemy cascade
    db.delete(conv)
    db.commit()

    if "application/json" in request.headers.get("accept", ""):
        return {"status": "ok", "deleted_id": conversation_id}

    return RedirectResponse(url="/conversations", status_code=status.HTTP_303_SEE_OTHER)


# --------------------------------------------------------------------------
# Answer Feedback Routes
# --------------------------------------------------------------------------

@router.post("/query/{query_id}/feedback")
async def submit_query_feedback(
    query_id: int,
    request: Request,
    rating: str = Form(...),
    reason: str = Form(None),
    comment: str = Form(None),
    workspace_id: str = Form(None),
    db: Session = Depends(get_db),
    user=Depends(get_current_user)
):
    ws_id = validate_workspace(workspace_id, user)

    # 1. Verify target query exists and is in user's workspace
    query_rec = db.query(QueryLog).filter(
        QueryLog.id == query_id,
        QueryLog.workspace_id == ws_id
    ).first()

    if not query_rec:
        raise HTTPException(status_code=404, detail="Query record not found or unauthorized")

    # 2. Validate rating server-side
    norm_rating = (rating or "").strip().lower()
    if norm_rating not in ALLOWED_RATINGS:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid rating '{rating}'. Allowed values are: {', '.join(sorted(ALLOWED_RATINGS))}"
        )

    # 3. Validate reason server-side
    norm_reason = (reason or "").strip().lower() if reason and reason.strip() else None
    if norm_reason and norm_reason not in ALLOWED_REASONS:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid reason '{reason}'. Allowed values are: {', '.join(sorted(ALLOWED_REASONS))}"
        )

    # 4. Clean comment
    norm_comment = (comment or "").strip()[:500] if comment else None

    # 5. Check if feedback already exists for (query_id, user_id)
    fb = db.query(QueryFeedback).filter(
        QueryFeedback.query_id == query_id,
        QueryFeedback.user_id == user.id
    ).first()

    if fb:
        # Update existing feedback record
        fb.rating = norm_rating
        fb.reason = norm_reason
        fb.comment = norm_comment
        fb.updated_at = datetime.utcnow()
    else:
        # Create new feedback record
        fb = QueryFeedback(
            query_id=query_id,
            conversation_id=query_rec.conversation_id,
            user_id=user.id,
            workspace_id=ws_id,
            rating=norm_rating,
            reason=norm_reason,
            comment=norm_comment,
            created_at=datetime.utcnow(),
            updated_at=datetime.utcnow()
        )
        db.add(fb)

    db.commit()
    db.refresh(fb)

    return {
        "status": "ok",
        "message": "Feedback recorded successfully",
        "feedback": {
            "id": fb.id,
            "query_id": fb.query_id,
            "rating": fb.rating,
            "reason": fb.reason,
            "comment": fb.comment,
            "updated_at": fb.updated_at.isoformat()
        }
    }


@router.get("/query/{query_id}/feedback")
async def get_query_feedback(
    query_id: int,
    workspace_id: str = None,
    db: Session = Depends(get_db),
    user=Depends(get_current_user)
):
    ws_id = validate_workspace(workspace_id, user)

    query_rec = db.query(QueryLog).filter(
        QueryLog.id == query_id,
        QueryLog.workspace_id == ws_id
    ).first()

    if not query_rec:
        raise HTTPException(status_code=404, detail="Query record not found or unauthorized")

    fb = db.query(QueryFeedback).filter(
        QueryFeedback.query_id == query_id,
        QueryFeedback.user_id == user.id
    ).first()

    if not fb:
        return {"feedback": None}

    return {
        "feedback": {
            "id": fb.id,
            "query_id": fb.query_id,
            "rating": fb.rating,
            "reason": fb.reason,
            "comment": fb.comment,
            "updated_at": fb.updated_at.isoformat()
        }
    }
